//! The GUI flow: the user's settings and the result of every stage, each cached
//! with the inputs that produced it, so changing a later stage's settings keeps the earlier results.
//!
//! Stages run as [`Job`]s — on the caller's thread ([`Session::ensure`], the wizard) or on a worker
//! (the GUI) with [`Session::accept`]
//! taking the result back.

use std::sync::Arc;

use vectorizeo_core::RgbaImage;
use vectorizeo_core::export::{ExportOptions, Format, ShapeMode, export};
use vectorizeo_core::params::{AdvancedSettings, EngineConfig, ImageType, Quality};
use vectorizeo_core::stages::classify::{
    Classification, ColorRecommendation, PaletteMode, classify, recommend_from_palettes,
};
use vectorizeo_core::stages::palette::{FoundPalette, PaletteParams, find_palettes_with_peaks};
use vectorizeo_core::stages::preprocess::{preprocess, shrink_for_analysis};
use vectorizeo_core::stages::segmenter::{Segmentation, segmentation_from_colors};

use crate::edit::{Background, SegEditor, zap_reference};
use crate::{Flatten, Vectorized, fit, flatten, palette_params, segment_with, trace_smooth};

/// A GUI progress stage (the label of each worker task).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stage {
    /// Image classification.
    Classify,
    /// Palette search.
    Palettes,
    /// Segmentation.
    Segment,
    /// Contour smoothing.
    Smooth,
    /// Bézier fitting.
    Fit,
}

impl Stage {
    pub fn label(self) -> &'static str {
        match self {
            Stage::Classify => "Classifying Image",
            Stage::Palettes => "Analyzing Image Colors",
            Stage::Segment => "Segmenting Image",
            Stage::Smooth => "Smoothing Result",
            Stage::Fit => "Reducing Nodes",
        }
    }
}

pub fn type_name(t: ImageType) -> &'static str {
    match t {
        ImageType::Photo => "Photograph",
        ImageType::ArtworkBlended => "Artwork with blended edges",
        ImageType::ArtworkUnblended => "Artwork without blending",
    }
}

pub fn quality_name(q: Quality) -> &'static str {
    match q {
        Quality::High => "High",
        Quality::Medium => "Medium",
        Quality::Low => "Low",
    }
}

/// Classification runs on load, on the image flattened to white.
/// `extension` as [`crate::extension`] gives it.
pub fn classify_loaded(loaded: &RgbaImage, extension: &str) -> Classification {
    classify(&flatten(loaded, Flatten::WHITE), extension)
}

#[derive(Clone, Debug, PartialEq)]
pub enum PaletteChoice {
    /// What the palette search recommends (the GUI preselects it).
    Recommended,
    Count(usize),
    Flexible,
    Custom(Vec<[u8; 4]>),
}

/// Advanced palette-finder settings (stage 1).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FinderSettings {
    /// `adv_color_sensitivity`: 0 → 16 bins, 6-connected peaks; 1 → 16 bins, 26-connected;
    /// 2 → 8 bins, 26-connected.
    pub sensitivity: i32,
    /// `adv_use_palette_aa`.
    pub anti_aliased: bool,
}

impl Default for FinderSettings {
    fn default() -> Self {
        Self { sensitivity: 1, anti_aliased: true }
    }
}

/// Everything the user can set, basic and advanced.
#[derive(Clone, Debug, PartialEq)]
pub struct Settings {
    pub advanced: bool,
    /// `flatten_image_mode`.
    pub flatten: Flatten,
    pub image_type: ImageType,
    pub quality: Quality,
    pub finder: FinderSettings,
    pub palette: PaletteChoice,
    pub adv: AdvancedSettings,
    /// Export option 0, `cake_stack_mode`.
    pub shape_mode: ShapeMode,
    /// `shape_stroking_mode == 0`.
    pub stroke: bool,
}

impl Settings {
    /// The state after loading an image: flatten to white (model reset), basic mode with
    /// the classifier's type and quality, the recommended palette, the default advanced settings
    /// and the export defaults.
    pub fn new(class: &Classification) -> Self {
        let export = ExportOptions::default();
        Self {
            advanced: false,
            flatten: Flatten::WHITE,
            image_type: class.image_type,
            quality: class.quality,
            finder: FinderSettings::default(),
            palette: PaletteChoice::Recommended,
            adv: AdvancedSettings::default(),
            shape_mode: export.mode,
            stroke: export.stroke,
        }
    }

    pub fn export_options(&self) -> ExportOptions {
        ExportOptions { mode: self.shape_mode, stroke: self.stroke }
    }

    /// The palette search parameters: the `pass1` config, anti-aliased iff blended artwork (basic) or as
    /// set (advanced, with the colour-sensitivity histogram settings).
    pub fn finder_params(&self) -> PaletteParams {
        if !self.advanced {
            return palette_params(self.image_type);
        }
        let mut p = PaletteParams::new(&EngineConfig::pass1(), self.finder.anti_aliased);
        (p.use_16_hist_bins, p.use_26_connected_peaks) = match self.finder.sensitivity {
            0 => (true, false),
            1 => (true, true),
            _ => (false, true),
        };
        p
    }

    pub fn palette_key(&self) -> PaletteKey {
        PaletteKey { flatten: self.flatten, params: self.finder_params() }
    }

    /// Whether the palette choice needs the palette search result.
    pub fn needs_palettes(&self) -> bool {
        matches!(self.palette, PaletteChoice::Recommended | PaletteChoice::Count(_))
    }

    /// The fixed palette to segment with, `None` for the flexible palette. `pals` must be the
    /// current palettes when [`Self::needs_palettes`].
    pub fn fixed_palette(&self, pals: Option<&Palettes>) -> Option<Vec<[u8; 4]>> {
        match &self.palette {
            PaletteChoice::Flexible => None,
            PaletteChoice::Custom(c) => Some(c.clone()),
            PaletteChoice::Count(k) => Some(pals.expect("palettes").found[k - 2].colors.clone()),
            PaletteChoice::Recommended => {
                let p = pals.expect("palettes");
                p.recommended().map(|i| p.found[i].colors.clone())
            }
        }
    }

    /// Segmentation and smoothing configs: the preset (basic), or the advanced base with the
    /// per-stage overrides of.
    pub fn configs(&self) -> (EngineConfig, EngineConfig) {
        if !self.advanced {
            let cfg = EngineConfig::preset(self.image_type, self.quality);
            return (cfg.clone(), cfg);
        }
        let mut seg = EngineConfig::advanced_base();
        seg.apply_advanced_segmentation(&self.adv);
        let mut smooth = EngineConfig::advanced_base();
        smooth.apply_advanced_smoothing(&self.adv);
        (seg, smooth)
    }
}

#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct PaletteKey {
    pub flatten: Flatten,
    pub params: PaletteParams,
}

/// The palette search 2..=12 and the colour-count recommendation for one input.
#[derive(Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Palettes {
    pub key: PaletteKey,
    pub found: Vec<FoundPalette>,
    pub rec: ColorRecommendation,
}

impl Palettes {
    pub fn find(loaded: &RgbaImage, key: PaletteKey) -> Self {
        // The palette search works on a copy shrunk to about 150 000 pixels.
        let img = shrink_for_analysis(&flatten(loaded, key.flatten), key.params.is_anti_aliased);
        let (found, peaks) = find_palettes_with_peaks(&img, &key.params, 2, 12);
        let rec = recommend_from_palettes(&img, &found, &peaks);
        Palettes { key, found, rec }
    }

    /// Index into `found` of the recommended fixed palette; `None` when flexible is recommended.
    pub fn recommended(&self) -> Option<usize> {
        (self.rec.mode != PaletteMode::Flexible).then(|| self.rec.colors - 2)
    }
}

#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct SegKey {
    pub flatten: Flatten,
    pub cfg: EngineConfig,
    pub palette: Option<Vec<[u8; 4]>>,
}

#[derive(Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct SegRun {
    pub key: SegKey,
    /// The preprocessed image the segmenter saw (smoothing measures against it).
    pub work: RgbaImage,
    pub seg: Segmentation,
}

impl SegRun {
    pub fn run(loaded: &RgbaImage, key: SegKey) -> Self {
        // The palette finder sees the flattened image; segmentation and smoothing the preprocessed one.
        let work = preprocess(&flatten(loaded, key.flatten), &key.cfg);
        let seg = segment_with(&work, &key.cfg, key.palette.as_deref());
        SegRun { key, work, seg }
    }
}

#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct VecKey {
    pub seg: SegKey,
    /// Smoothing/fitting config.
    pub cfg: EngineConfig,
    /// [`SegEditor::revision`] of the manual segmentation edits, if any.
    pub edit: Option<u64>,
}

#[derive(Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct VecRun {
    pub key: VecKey,
    /// The segmentation that was vectorized (edited, if there are edits).
    pub seg: Arc<SegRun>,
    pub v: Vectorized,
}

/// How far the pipeline has to run.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Target {
    Palettes,
    Segmentation,
    Result,
}

/// One stale stage with its inputs; [`Job::run`] works on any thread.
#[derive(Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum Job {
    Palettes {
        loaded: Arc<RgbaImage>,
        key: PaletteKey,
    },
    Segment {
        loaded: Arc<RgbaImage>,
        key: SegKey,
    },
    /// `edit`: the edited colour image, committed before tracing.
    Vectorize {
        seg: Arc<SegRun>,
        edit: Option<Arc<Vec<[u8; 4]>>>,
        key: VecKey,
    },
}

/// A finished [`Job`].
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum Done {
    Palettes(Arc<Palettes>),
    Segment(Arc<SegRun>),
    Vectorize(Arc<VecRun>),
}

impl Job {
    /// Whether both jobs compute the same result (same stage and inputs).
    pub fn same_as(&self, other: &Job) -> bool {
        match (self, other) {
            (Job::Palettes { key: a, .. }, Job::Palettes { key: b, .. }) => a == b,
            (Job::Segment { key: a, .. }, Job::Segment { key: b, .. }) => a == b,
            (Job::Vectorize { key: a, .. }, Job::Vectorize { key: b, .. }) => a == b,
            _ => false,
        }
    }

    pub fn first_stage(&self) -> Stage {
        match self {
            Job::Palettes { .. } => Stage::Palettes,
            Job::Segment { .. } => Stage::Segment,
            Job::Vectorize { .. } => Stage::Smooth,
        }
    }

    /// Run the job; `on_stage` is called as each stage begins (vectorizing is two stages).
    pub fn run(self, on_stage: &mut dyn FnMut(Stage)) -> Done {
        on_stage(self.first_stage());
        match self {
            Job::Palettes { loaded, key } => Done::Palettes(Arc::new(Palettes::find(&loaded, key))),
            Job::Segment { loaded, key } => Done::Segment(Arc::new(SegRun::run(&loaded, key))),
            Job::Vectorize { seg, edit, key } => {
                let seg = match edit {
                    Some(img) => Arc::new(SegRun {
                        key: seg.key.clone(),
                        seg: segmentation_from_colors(&seg.work, &img),
                        work: seg.work.clone(),
                    }),
                    None => seg,
                };
                let (graph, colors) = trace_smooth(&seg.seg, &seg.work, &key.cfg, true);
                on_stage(Stage::Fit);
                let curves = Some(fit(&graph, &key.cfg));
                Done::Vectorize(Arc::new(VecRun { key, seg, v: Vectorized { graph, curves, colors } }))
            }
        }
    }
}

/// One loaded image, its settings and the latest result of every stage.
pub struct Session {
    /// Premultiplied, as the engine loads it.
    pub loaded: Arc<RgbaImage>,
    pub has_alpha: bool,
    pub class: Classification,
    pub settings: Settings,
    palettes: Option<Arc<Palettes>>,
    seg: Option<Arc<SegRun>>,
    result: Option<Arc<VecRun>>,
    /// Edit Segmentation, on the segmentation it was fetched from.
    seg_edit: Option<(Arc<SegRun>, SegEditor)>,
    /// Remove Background, for `result`.
    background: Background,
}

impl Session {
    pub fn new(loaded: RgbaImage, class: Classification) -> Self {
        let has_alpha = loaded.pixels.iter().any(|p| p.a < 255);
        let settings = Settings::new(&class);
        Self {
            loaded: Arc::new(loaded),
            has_alpha,
            class,
            settings,
            palettes: None,
            seg: None,
            result: None,
            seg_edit: None,
            background: Background::default(),
        }
    }

    /// The palettes for the current settings, if found.
    pub fn palettes(&self) -> Option<&Arc<Palettes>> {
        self.palettes.as_ref().filter(|p| p.key == self.settings.palette_key())
    }

    /// The segmentation key for the current settings; `None` while the palettes it needs are missing.
    pub fn seg_key(&self) -> Option<SegKey> {
        let pals = self.palettes();
        if self.settings.needs_palettes() && pals.is_none() {
            return None;
        }
        let palette = self.settings.fixed_palette(pals.map(|p| &**p));
        Some(SegKey { flatten: self.settings.flatten, cfg: self.settings.configs().0, palette })
    }

    pub fn vec_key(&self) -> Option<VecKey> {
        let edit = self.seg_edits().map(SegEditor::revision);
        Some(VecKey { seg: self.seg_key()?, cfg: self.settings.configs().1, edit })
    }

    /// The segmentation editor for the current segmentation, if one was opened.
    pub fn seg_editor(&self) -> Option<&SegEditor> {
        let cur = self.segmentation()?;
        self.seg_edit.as_ref().filter(|(s, _)| Arc::ptr_eq(s, cur)).map(|(_, e)| e)
    }

    /// The editor with edits in it, for the current segmentation.
    fn seg_edits(&self) -> Option<&SegEditor> {
        self.seg_editor().filter(|e| e.has_edits())
    }

    /// The segmentation editor, fetched from the current segmentation if it is not open on it
    /// (a new segmentation replaces the edits).
    pub fn seg_editor_mut(&mut self) -> Option<&mut SegEditor> {
        let cur = self.segmentation()?.clone();
        if !self.seg_edit.as_ref().is_some_and(|(s, _)| Arc::ptr_eq(s, &cur)) {
            let reference = zap_reference(&self.loaded, cur.key.flatten);
            let editor = SegEditor::new(&cur.seg, reference);
            self.seg_edit = Some((cur, editor));
        }
        self.seg_edit.as_mut().map(|(_, e)| e)
    }

    /// Whether the current settings need a new segmentation although the last one has manual
    /// edits (the GUI asks before overwriting them).
    pub fn settings_discard_edits(&self) -> bool {
        let Some((s, e)) = &self.seg_edit else { return false };
        e.has_edits() && self.seg_key().as_ref() != Some(&s.key)
    }

    pub fn discard_seg_edits(&mut self) {
        self.seg_edit = None;
    }

    /// Remove Background state of the current result.
    pub fn background(&self) -> &Background {
        &self.background
    }

    pub fn background_mut(&mut self) -> &mut Background {
        &mut self.background
    }

    /// The segmentation for the current settings, if done.
    pub fn segmentation(&self) -> Option<&Arc<SegRun>> {
        let key = self.seg_key()?;
        self.seg.as_ref().filter(|s| s.key == key)
    }

    /// The vector result for the current settings, if done.
    pub fn result(&self) -> Option<&Arc<VecRun>> {
        let key = self.vec_key()?;
        self.result.as_ref().filter(|r| r.key == key)
    }

    /// The latest segmentation and result, possibly for older settings.
    pub fn last_segmentation(&self) -> Option<&Arc<SegRun>> {
        self.seg.as_ref()
    }

    pub fn last_result(&self) -> Option<&Arc<VecRun>> {
        self.result.as_ref()
    }

    /// The first stage that is stale for `target`, with its inputs; `None` when `target` is current.
    pub fn next_job(&self, target: Target) -> Option<Job> {
        if (target == Target::Palettes || self.settings.needs_palettes()) && self.palettes().is_none() {
            return Some(Job::Palettes { loaded: self.loaded.clone(), key: self.settings.palette_key() });
        }
        if target == Target::Palettes {
            return None;
        }
        let Some(seg) = self.segmentation() else {
            return Some(Job::Segment { loaded: self.loaded.clone(), key: self.seg_key()? });
        };
        if target == Target::Segmentation || self.result().is_some() {
            return None;
        }
        let edit = self.seg_edits().map(|e| e.image().clone());
        Some(Job::Vectorize { seg: seg.clone(), edit, key: self.vec_key()? })
    }

    /// Keep a finished job's result (whether or not it still matches the settings).
    pub fn accept(&mut self, done: Done) {
        match done {
            Done::Palettes(p) => self.palettes = Some(p),
            Done::Segment(s) => self.seg = Some(s),
            Done::Vectorize(r) => {
                // Every finished vectorization resets Remove Background.
                self.background = Background::new(r.v.colors.len());
                self.result = Some(r);
            }
        }
    }

    /// Run every stale stage for `target` on this thread.
    pub fn ensure(&mut self, target: Target, on_stage: &mut dyn FnMut(Stage)) {
        while let Some(job) = self.next_job(target) {
            let done = job.run(on_stage);
            self.accept(done);
        }
    }

    /// Export options at save time: stacked shapes become cut-outs when the image
    /// has transparency or shapes were removed (the Remove Background history is not empty).
    pub fn save_options(&self) -> ExportOptions {
        let mut o = self.settings.export_options();
        if o.mode == ShapeMode::Stacked && (self.has_alpha || self.background.has_edits()) {
            o.mode = ShapeMode::CutOuts;
        }
        o
    }

    /// The SVG of the current result as it is saved: export options, removed shapes skipped.
    pub fn svg(&self) -> Option<String> {
        self.export(Format::Svg)
    }

    /// The current result as it is saved, in `format`.
    pub fn export(&self, format: Format) -> Option<String> {
        let r = &self.result()?.v;
        let hidden = self.background.hidden();
        Some(export(&r.graph, r.curves.as_ref(), &r.colors, self.save_options(), hidden, format))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{palette_input, vectorize};
    use vectorizeo_core::Rgba;
    use vectorizeo_core::export::export_svg;

    /// White with a blue disc and a red square.
    fn sample() -> RgbaImage {
        let (w, h) = (40, 32);
        let mut pixels = Vec::with_capacity(w * h);
        for y in 0..h {
            for x in 0..w {
                let (dx, dy) = (x as f64 - 14.0, y as f64 - 16.0);
                let p = if dx * dx + dy * dy < 81.0 {
                    Rgba { r: 20, g: 4, b: 171, a: 255 }
                } else if (26..36).contains(&x) && (6..16).contains(&y) {
                    Rgba { r: 220, g: 30, b: 30, a: 255 }
                } else {
                    Rgba { r: 255, g: 255, b: 255, a: 255 }
                };
                pixels.push(p);
            }
        }
        RgbaImage::new(w, h, pixels)
    }

    /// With every default accepted the session gives `vectorizeo trace -c auto`'s SVG.
    #[test]
    fn defaults_match_trace_auto() {
        let loaded = sample();
        let class = classify_loaded(&loaded, ".png");
        let mut s = Session::new(loaded.clone(), class.clone());
        s.ensure(Target::Result, &mut |_| {});
        let ours = s.svg().unwrap();

        let img = flatten(&loaded, Flatten::WHITE);
        let cfg = EngineConfig::preset(class.image_type, class.quality);
        let work = preprocess(&img, &cfg);
        let small = palette_input(&img, class.image_type);
        let (mut pals, peaks) = find_palettes_with_peaks(&small, &palette_params(class.image_type), 2, 12);
        let rec = recommend_from_palettes(&small, &pals, &peaks);
        let palette = (rec.mode != PaletteMode::Flexible).then(|| pals.swap_remove(rec.colors - 2).colors);
        let seg = segment_with(&work, &cfg, palette.as_deref());
        let v = vectorize(&seg, &work, &cfg, true, true);
        assert_eq!(ours, export_svg(&v.graph, v.curves.as_ref(), &v.colors, ExportOptions::default()));
    }

    /// Changing a smoothing setting keeps the segmentation; changing the palette does not.
    #[test]
    fn stale_stages() {
        let loaded = sample();
        let class = classify_loaded(&loaded, ".png");
        let mut s = Session::new(loaded, class);
        s.settings.advanced = true;
        s.settings.palette = PaletteChoice::Flexible;
        assert!(matches!(s.next_job(Target::Result), Some(Job::Segment { .. })));
        s.ensure(Target::Result, &mut |_| {});
        assert!(s.next_job(Target::Result).is_none());

        s.settings.adv.smoothness += 1;
        assert!(s.segmentation().is_some());
        assert!(matches!(s.next_job(Target::Result), Some(Job::Vectorize { .. })));
        let mut stages = Vec::new();
        s.ensure(Target::Result, &mut |st| stages.push(st));
        assert_eq!(stages, [Stage::Smooth, Stage::Fit]);

        s.settings.shape_mode = ShapeMode::Stacked;
        assert!(s.next_job(Target::Result).is_none());

        s.settings.palette = PaletteChoice::Recommended;
        assert!(matches!(s.next_job(Target::Segmentation), Some(Job::Palettes { .. })));
    }

    /// Jobs and their results survive a serialization round trip (the web GUI's Web Worker).
    #[cfg(feature = "serde")]
    #[test]
    fn jobs_round_trip() {
        let loaded = sample();
        let class = classify_loaded(&loaded, ".png");
        let mut direct = Session::new(loaded.clone(), class.clone());
        direct.ensure(Target::Result, &mut |_| {});
        let mut s = Session::new(loaded, class);
        while let Some(job) = s.next_job(Target::Result) {
            let job: Job = postcard::from_bytes(&postcard::to_allocvec(&job).unwrap()).unwrap();
            let done = job.run(&mut |_| {});
            let done: Done = postcard::from_bytes(&postcard::to_allocvec(&done).unwrap()).unwrap();
            s.accept(done);
        }
        assert_eq!(s.svg(), direct.svg());
    }

    /// Edits re-run the vectorization on the committed colours; a settings change that
    /// re-segments is flagged; Remove Background drops the shape from the SVG.
    #[test]
    fn edits_flow_into_the_result() {
        use crate::edit::Action;
        let loaded = sample();
        let class = classify_loaded(&loaded, ".png");
        let mut s = Session::new(loaded, class);
        s.ensure(Target::Result, &mut |_| {});
        let before = s.result().unwrap().clone();
        let c = before.v.colors[before.seg.seg.labels[10 * 40 + 30] as usize];
        let red = format!("#{:02x}{:02x}{:02x}", c[0], c[1], c[2]);
        assert!(s.svg().unwrap().contains(&red));

        // Paint the red square white: one segment fewer.
        let white = s.seg_editor_mut().unwrap().color_at(0, 0).unwrap();
        assert!(s.seg_editor_mut().unwrap().apply(Action::Fill { x: 30, y: 10, color: white }));
        assert!(matches!(s.next_job(Target::Result), Some(Job::Vectorize { edit: Some(_), .. })));
        s.ensure(Target::Result, &mut |_| {});
        let after = s.result().unwrap().clone();
        assert_eq!(after.seg.seg.segments.len() + 1, before.seg.seg.segments.len());
        assert!(!s.svg().unwrap().contains(&red));

        // Undo everything: back to the unedited segmentation's key.
        s.seg_editor_mut().unwrap().undo();
        assert_eq!(s.vec_key().unwrap(), before.key);
        assert!(!s.settings_discard_edits());
        s.seg_editor_mut().unwrap().redo();
        s.settings.flatten = Flatten::from_hex("000000").unwrap();
        assert!(s.settings_discard_edits());
        s.settings.flatten = Flatten::WHITE;

        // Remove the disc.
        s.ensure(Target::Result, &mut |_| {});
        let r = s.result().unwrap().clone();
        let disc = r.v.graph.shape_at(&r.seg.seg.labels, 14.5, 16.5).unwrap();
        let svg = s.svg().unwrap();
        s.background_mut().toggle(disc);
        let removed = s.svg().unwrap();
        assert_eq!(removed.matches("<path").count() + 1, svg.matches("<path").count());
    }
}
