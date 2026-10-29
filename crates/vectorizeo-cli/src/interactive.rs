//! `vectorizeo interactive`: the GUI's step-by-step flow in the terminal (basic and advanced mode),
//! with the input, palettes, segmentation and result shown inline.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Instant;

use vectorizeo_core::export::ShapeMode;
use vectorizeo_core::params::{AaRejection, ImageType, Quality};
use vectorizeo_pipeline::session::{PaletteChoice, Session, Stage, Target, classify_loaded, quality_name, type_name};
use vectorizeo_pipeline::{
    Flatten, extension, hex_rgba, load, parse_colors, preview_scale, render_svg, segmentation_image,
};

use crate::term::{Nav, ask, ask_bool, ask_int, choose, heading, info, show_image, swatches};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Step {
    Transparency,
    Mode,
    ImageType,
    Quality,
    PaletteFinder,
    Palette,
    Segmentation,
    Smoothing,
    Segment,
    Result,
    Save,
}

struct Wizard {
    output: PathBuf,
    /// The file as loaded (straight alpha), for display.
    original: image::RgbaImage,
    s: Session,
}

/// Prints each stage's GUI progress label and the time it took.
#[derive(Default)]
struct Timer(Option<Instant>);

impl Timer {
    fn stage(&mut self, stage: Stage) {
        self.finish();
        print!("{}… ", stage.label());
        let _ = std::io::stdout().flush();
        self.0 = Some(Instant::now());
    }

    fn finish(&mut self) {
        if let Some(t) = self.0.take() {
            println!("done ({:.1} s)", t.elapsed().as_secs_f64());
        }
    }
}

impl Wizard {
    fn new(input: &Path, output: Option<PathBuf>) -> Result<Self, String> {
        let original = image::open(input).map_err(|e| format!("cannot open {}: {e}", input.display()))?.to_rgba8();
        let loaded = load(input)?;
        let has_alpha = loaded.pixels.iter().any(|p| p.a < 255);
        heading(&format!("Vectorizeo — {}", input.display()));
        println!("{} × {} pixels{}", loaded.width, loaded.height, if has_alpha { ", with transparency" } else { "" });
        show_image(&original);
        let mut timer = Timer::default();
        timer.stage(Stage::Classify);
        let class = classify_loaded(&loaded, &extension(input));
        timer.finish();
        println!(
            "Classified as {}, {} quality ({})",
            type_name(class.image_type),
            quality_name(class.quality),
            class.label
        );
        // Saved in the current directory under the input's name.
        let output =
            output.unwrap_or_else(|| PathBuf::from(input.file_stem().unwrap_or_default()).with_extension("svg"));
        Ok(Self { output, original, s: Session::new(loaded, class) })
    }

    fn run(&mut self) -> Result<(), String> {
        let mut history: Vec<Step> = Vec::new();
        let mut step = if self.s.has_alpha { Step::Transparency } else { Step::Mode };
        loop {
            match self.step(step, !history.is_empty()) {
                Ok(Some(next)) => {
                    history.push(step);
                    step = next;
                }
                Ok(None) | Err(Nav::Quit) => return Ok(()),
                Err(Nav::Back) => {
                    if let Some(prev) = history.pop() {
                        step = prev;
                    }
                }
            }
        }
    }

    /// One wizard page; returns the next step, or `None` when done.
    fn step(&mut self, step: Step, can_back: bool) -> Result<Option<Step>, Nav> {
        let next = match step {
            Step::Transparency => {
                let custom = match self.s.settings.flatten {
                    Flatten(Some(c)) if c != [1.0; 4] => c,
                    _ => [1.0; 4],
                };
                let default = match self.s.settings.flatten {
                    Flatten(None) => 2,
                    Flatten(Some(c)) if c != [1.0; 4] => 1,
                    _ => 0,
                };
                let options = ["Flatten onto white", "Flatten onto a colour…", "Leave transparent"].map(String::from);
                self.s.settings.flatten = match choose("Transparency", &options, default, can_back)? {
                    0 => Flatten::WHITE,
                    1 => {
                        let hex = |c: f32| (c * 255.0).round() as u8;
                        let def = format!("{:02x}{:02x}{:02x}", hex(custom[0]), hex(custom[1]), hex(custom[2]));
                        ask("Background colour RRGGBB", &def, |s| {
                            Flatten::from_hex(s).ok_or_else(|| "expected RRGGBB".into())
                        })?
                    }
                    _ => Flatten(None),
                };
                Step::Mode
            }
            Step::Mode => {
                let options = [
                    "Basic — pick image type, quality and palette",
                    "Advanced — set the segmentation and smoothing controls",
                ]
                .map(String::from);
                self.s.settings.advanced =
                    choose("Mode", &options, usize::from(self.s.settings.advanced), can_back)? == 1;
                if self.s.settings.advanced { Step::PaletteFinder } else { Step::ImageType }
            }
            Step::ImageType => {
                let types = [ImageType::Photo, ImageType::ArtworkBlended, ImageType::ArtworkUnblended];
                let options = types.map(|t| {
                    let mark = if t == self.s.class.image_type { "  (recommended)" } else { "" };
                    format!("{}{mark}", type_name(t))
                });
                let default = types.iter().position(|&t| t == self.s.settings.image_type).unwrap_or(0);
                self.s.settings.image_type = types[choose("Image type", &options, default, can_back)?];
                Step::Quality
            }
            Step::Quality => {
                let levels = [Quality::High, Quality::Medium, Quality::Low];
                let title =
                    if self.s.settings.image_type == ImageType::Photo { "Detail level" } else { "Image quality" };
                let options = levels.map(|q| {
                    let mark = if q == self.s.class.quality { "  (recommended)" } else { "" };
                    format!("{}{mark}", quality_name(q))
                });
                let default = levels.iter().position(|&q| q == self.s.settings.quality).unwrap_or(0);
                self.s.settings.quality = levels[choose(title, &options, default, can_back)?];
                Step::Palette
            }
            Step::PaletteFinder => {
                heading("Palette finder");
                let f = &mut self.s.settings.finder;
                info("Colour sensitivity: 0 = 16 bins, 6-connected peaks; 1 = 16 bins, 26-connected; 2 = 8 bins.");
                f.sensitivity = ask_int("Colour sensitivity", 0, 2, f.sensitivity)?;
                f.anti_aliased = ask_bool("Anti-aliased edges (palette)", f.anti_aliased)?;
                Step::Palette
            }
            Step::Palette => {
                self.palette_step(can_back)?;
                if self.s.settings.advanced { Step::Segmentation } else { Step::Segment }
            }
            Step::Segmentation => {
                heading("Segmentation");
                let a = &mut self.s.settings.adv;
                a.segmentation_complexity =
                    ask_int("Segmentation complexity (more = more segments)", 1, 12, a.segmentation_complexity)?;
                a.min_num_pixels = ask_int("Minimum segment size in pixels", 0, 1000, a.min_num_pixels)?;
                let level = ask_int("Anti-aliasing rejection level", 0, 2, a.aa_rejection as i32)?;
                a.aa_rejection = [AaRejection::Level0, AaRejection::Level1, AaRejection::Level2][level as usize];
                a.cluster_colors = ask_bool("Cluster similar colours", a.cluster_colors)?;
                Step::Smoothing
            }
            Step::Smoothing => {
                heading("Smoothing");
                let a = &mut self.s.settings.adv;
                a.smoothness = ask_int("Contour smoothness", 1, 12, a.smoothness)?;
                a.anti_aliasing_engine = ask_bool("Anti-aliasing engine", a.anti_aliasing_engine)?;
                a.detect_sharp_corners = ask_bool("Detect sharp corners", a.detect_sharp_corners)?;
                a.num_nodes = ask_int("Curve complexity (number of nodes)", 1, 12, a.num_nodes)?;
                Step::Segment
            }
            Step::Segment => {
                self.ensure(Target::Segmentation);
                let run = self.s.segmentation().expect("segmented").clone();
                heading("Segmentation result");
                println!("{} segments", run.seg.segments.len());
                show_image(&segmentation_image(&run.seg));
                let options = ["Continue: smooth and fit curves", "Back to the settings"].map(String::from);
                if choose("Next", &options, 0, can_back)? == 1 {
                    return Err(Nav::Back);
                }
                Step::Result
            }
            Step::Result => {
                self.ensure(Target::Result);
                let run = self.s.result().expect("vectorized").clone();
                let svg = self.s.svg().expect("vectorized");
                heading("Result");
                let curves = run.v.curves.as_ref().map_or(0, |c| c.fits.len());
                println!("{} shapes, {} curves", run.v.graph.contours.len(), curves);
                if let Some(img) = render_svg(&svg, preview_scale(self.s.loaded.width, self.s.loaded.height)) {
                    show_image(&img);
                }
                let options =
                    ["Save", "Compare with the input", "Back to the settings", "Quit without saving"].map(String::from);
                loop {
                    match choose("Next", &options, 0, can_back)? {
                        0 => break Step::Save,
                        1 => show_image(&self.original),
                        2 => return Err(Nav::Back),
                        _ => return Err(Nav::Quit),
                    }
                }
            }
            Step::Save => {
                self.save_step(can_back)?;
                let options = ["Try other settings", "Quit"].map(String::from);
                match choose("Done", &options, 1, false)? {
                    0 => Step::Mode,
                    _ => return Ok(None),
                }
            }
        };
        Ok(Some(next))
    }

    /// Run every stale stage for `target`, printing the stage labels.
    fn ensure(&mut self, target: Target) {
        let mut timer = Timer::default();
        self.s.ensure(target, &mut |stage| timer.stage(stage));
        timer.finish();
    }

    fn palette_step(&mut self, can_back: bool) -> Result<(), Nav> {
        self.ensure(Target::Palettes);
        let pals = self.s.palettes().expect("palettes found").clone();
        let rec = pals.recommended();
        let mut options: Vec<String> = pals
            .found
            .iter()
            .enumerate()
            .map(|(i, p)| {
                let mark = if Some(i) == rec { "  (recommended)" } else { "" };
                format!("{:>2} colours  {}{mark}", i + 2, swatches(&p.colors))
            })
            .collect();
        let flexible = options.len();
        let mark = if rec.is_none() { "  (recommended)" } else { "" };
        options.push(format!("Flexible palette (colours found during segmentation){mark}"));
        options.push("Custom colours…".into());
        let default = match &self.s.settings.palette {
            PaletteChoice::Recommended => rec.unwrap_or(flexible),
            PaletteChoice::Count(k) => k - 2,
            PaletteChoice::Flexible => flexible,
            PaletteChoice::Custom(_) => flexible + 1,
        };
        let i = choose("Palette", &options, default, can_back)?;
        self.s.settings.palette = if i < flexible {
            if Some(i) == rec { PaletteChoice::Recommended } else { PaletteChoice::Count(i + 2) }
        } else if i == flexible {
            PaletteChoice::Flexible
        } else {
            let def = match &self.s.settings.palette {
                PaletteChoice::Custom(c) => {
                    c.iter().map(|&c| hex_rgba(c)[1..].to_string()).collect::<Vec<_>>().join(",")
                }
                _ => pals.found[rec.unwrap_or(0)]
                    .colors
                    .iter()
                    .map(|&c| hex_rgba(c)[1..7].to_string())
                    .collect::<Vec<_>>()
                    .join(","),
            };
            let colors = ask("Colours RRGGBB[AA],…", &def, parse_colors)?;
            println!("{}", swatches(&colors));
            PaletteChoice::Custom(colors)
        };
        Ok(())
    }

    fn save_step(&mut self, can_back: bool) -> Result<(), Nav> {
        let modes = [ShapeMode::Stacked, ShapeMode::CutOuts, ShapeMode::CutOutsGrouped];
        let options = [
            "Stacked — shapes on top of each other",
            "Cut-outs — shapes in holes of the shapes below",
            "Cut-outs, grouped by colour",
        ]
        .map(String::from);
        let default = modes.iter().position(|&m| m == self.s.settings.shape_mode).unwrap_or(2);
        self.s.settings.shape_mode = modes[choose("Shape mode", &options, default, can_back)?];
        self.s.settings.stroke = ask_bool("Stroke shapes with a hairline", self.s.settings.stroke)?;
        loop {
            let path: PathBuf = ask("Save as", &self.output.display().to_string(), |s| Ok(PathBuf::from(s)))?;
            if path.exists()
                && path != self.output
                && !ask_bool(&format!("{} exists; overwrite?", path.display()), false)?
            {
                continue;
            }
            self.ensure(Target::Result);
            let doc = self.s.export(crate::output_format(&path)).expect("vectorized");
            match std::fs::write(&path, doc) {
                Ok(()) => {
                    println!("Saved {}", path.display());
                    self.output = path;
                    return Ok(());
                }
                Err(e) => println!("cannot write {}: {e}", path.display()),
            }
        }
    }
}

/// Entry point of `vectorizeo interactive`.
pub fn run(input: &Path, output: Option<PathBuf>) -> Result<(), String> {
    use std::io::IsTerminal;
    if !std::io::stdin().is_terminal() || !std::io::stdout().is_terminal() {
        return Err("vectorizeo interactive needs a terminal; use `vectorizeo trace` in scripts".into());
    }
    Wizard::new(input, output)?.run()
}
