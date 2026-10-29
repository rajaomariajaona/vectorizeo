//! Main window: top bar (open, views, compare, edit, export), settings panel on the right, status
//! bar, and the canvas. Stages run on a [`Worker`]; results are cached per stage in the [`Session`].

use std::path::PathBuf;
use std::sync::Arc;
use std::sync::mpsc::{Receiver, Sender, channel};
use std::time::Duration;

use eframe::egui::{
    self, Align, Color32, ColorImage, CornerRadius, Key, KeyboardShortcut, Layout, Margin, Modifiers, RichText, Stroke,
    TextureHandle, TextureOptions, ThemePreference, Ui, pos2, vec2,
};
use vectorizeo_core::export::{ExportOptions, Format, export_svg_hidden, outline};
use vectorizeo_gui::task::{Image, Loaded, Output, Task};
use vectorizeo_pipeline::session::{Job, SegRun, Session, Settings, Target, VecRun, quality_name, type_name};
use vectorizeo_pipeline::{Flatten, segmentation_image};
use web_time::Instant;

use crate::canvas::{Canvas, Layer};
use crate::edit::EditTools;
use crate::inspector::{self, Hover};
use crate::theme::{self, accent_button, accent_split_button, caption, segmented};
use crate::worker::Worker;

/// Pieces per cubic in the removed-shape outlines.
const OUTLINE_STEPS: usize = 12;
const OPEN: KeyboardShortcut = KeyboardShortcut::new(Modifiers::COMMAND, Key::O);
const SAVE: KeyboardShortcut = KeyboardShortcut::new(Modifiers::COMMAND, Key::S);
/// How long the "Saved …" notice stays.
const TOAST: Duration = Duration::from_secs(4);
const IMAGE_EXTENSIONS: [&str; 7] = ["png", "jpg", "jpeg", "bmp", "gif", "tif", "tiff"];
/// Quiet time after a settings change before the stages re-run (slider drags).
const DEBOUNCE: Duration = Duration::from_millis(250);
/// Largest side of a rendered result texture.
const MAX_RENDER: f32 = 4096.0;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum View {
    Input,
    Segments,
    Result,
}

/// What a result texture shows: a result, exported with some options and without the removed
/// shapes, rasterised at a scale.
#[derive(Clone)]
struct RenderKey {
    result: Arc<VecRun>,
    export: ExportOptions,
    hidden: Vec<bool>,
    scale: f32,
}

impl RenderKey {
    /// The same picture, at any scale.
    fn same_content(&self, o: &Self) -> bool {
        Arc::ptr_eq(&self.result, &o.result)
            && self.export.mode == o.export.mode
            && self.export.stroke == o.export.stroke
            && self.hidden == o.hidden
    }
}

impl PartialEq for RenderKey {
    fn eq(&self, o: &Self) -> bool {
        self.same_content(o) && self.scale == o.scale
    }
}

/// A hovered option and the flatten setting it is previewed with.
type PreviewKey = (Hover, Flatten);

/// The open image.
struct Document {
    path: PathBuf,
    /// As loaded (straight alpha), for display.
    original: image::RgbaImage,
    input_tex: TextureHandle,
    session: Session,
    /// Settings when the stages were last scheduled (change detection).
    last_settings: Settings,
    /// The segments texture: the segmentation, or its edited colours at an editor revision.
    seg_tex: Option<(Arc<SegRun>, Option<u64>, TextureHandle)>,
    result_tex: Option<(RenderKey, TextureHandle)>,
    pending_render: Option<RenderKey>,
    /// Outlines of the removed shapes of a result at a Remove Background revision.
    removed: Option<(Arc<VecRun>, u64, Vec<Vec<[f64; 2]>>)>,
}

impl Document {
    /// The segments texture is of the current segmentation (edits update it in place).
    fn seg_fresh(&self) -> bool {
        match (self.session.segmentation(), &self.seg_tex) {
            (Some(s), Some((t, _, _))) => Arc::ptr_eq(s, t),
            _ => false,
        }
    }

    /// The result texture to show for the current state (any scale).
    fn render_key(&self, scale: f32) -> Option<RenderKey> {
        let result = self.session.result()?.clone();
        let hidden = self.session.background().hidden().to_vec();
        Some(RenderKey { result, export: self.session.save_options(), hidden, scale })
    }

    /// The result texture shows the current result with the current export options; removed
    /// shapes may lag a frame behind (their outlines are drawn on top right away).
    fn result_fresh(&self) -> bool {
        match (self.render_key(1.0), &self.result_tex) {
            (Some(want), Some((k, _))) => {
                Arc::ptr_eq(&want.result, &k.result)
                    && want.export.mode == k.export.mode
                    && want.export.stroke == k.export.stroke
            }
            _ => false,
        }
    }
}

pub struct App {
    doc: Option<Document>,
    canvas: Canvas,
    view: View,
    compare: bool,
    error: Option<String>,
    worker: Worker,
    /// Rasterises results for `Document::pending_render`.
    render: Worker,
    /// The stage job on `worker`, if any (loading is not a job).
    running: Option<Job>,
    /// Re-run stale stages as soon as settings change.
    auto_update: bool,
    /// Run the stale stages once although auto update is off.
    update_requested: bool,
    /// Set by Cancel: nothing re-runs until the settings change or Update is clicked.
    paused: bool,
    changed_at: Option<Instant>,
    /// The option under the pointer in the settings panel.
    hover: Option<Hover>,
    /// Computes the preview for `pending_preview`.
    previews: Worker,
    preview: Option<(PreviewKey, TextureHandle)>,
    pending_preview: Option<PreviewKey>,
    /// Export asked for while the result was still being computed.
    export_pending: bool,
    /// Outcome of the last save: message, success, when.
    toast: Option<(String, bool, Instant)>,
    edit: EditTools,
    /// Format of the last export (the save dialog's default).
    export_format: Format,
    /// Settings before a change that would discard the segmentation edits, while asking.
    confirm: Option<Settings>,
    /// Files picked or dropped in the browser (read asynchronously).
    picked: (Sender<Source>, Receiver<Source>),
}

/// Textures keep pixels sharp when zoomed in and smooth when zoomed out.
const TEXTURE_OPTIONS: TextureOptions =
    TextureOptions { magnification: egui::TextureFilter::Nearest, ..TextureOptions::LINEAR };

fn to_color_image(img: &image::RgbaImage) -> ColorImage {
    ColorImage::from_rgba_unmultiplied([img.width() as usize, img.height() as usize], img.as_raw())
}

fn image_to_color(img: &Image) -> ColorImage {
    ColorImage::from_rgba_unmultiplied([img.width as usize, img.height as usize], &img.rgba)
}

/// An image to open: a file, or a file name with its contents (browser picks and drops).
struct Source {
    path: PathBuf,
    bytes: Option<Arc<[u8]>>,
}

impl App {
    pub fn new(cc: &eframe::CreationContext<'_>, input: Option<PathBuf>) -> Self {
        let picked = channel();
        theme::install(&cc.egui_ctx);
        cc.egui_ctx.set_theme(ThemePreference::System);
        let mut app = Self {
            doc: None,
            canvas: Canvas::default(),
            view: View::Input,
            compare: false,
            error: None,
            worker: Worker::new(&cc.egui_ctx),
            render: Worker::new(&cc.egui_ctx),
            running: None,
            auto_update: true,
            update_requested: false,
            paused: false,
            changed_at: None,
            hover: None,
            previews: Worker::new(&cc.egui_ctx),
            preview: None,
            pending_preview: None,
            export_pending: false,
            toast: None,
            edit: EditTools::default(),
            export_format: Format::Svg,
            confirm: None,
            picked,
        };
        if let Some(path) = input {
            app.open(Source { path, bytes: None });
        }
        app
    }

    fn open(&mut self, src: Source) {
        self.running = None;
        self.error = None;
        self.worker.spawn(Task::Load { path: src.path, bytes: src.bytes.map(|b| b.to_vec()) });
    }

    fn set_document(&mut self, ctx: &egui::Context, l: Loaded) {
        let original = image::RgbaImage::from_raw(l.original.width, l.original.height, l.original.rgba)
            .expect("width × height pixels");
        let input_tex = ctx.load_texture("input", to_color_image(&original), TEXTURE_OPTIONS);
        self.canvas.reset(original.width() as usize, original.height() as usize);
        let name = l.path.file_name().unwrap_or_default().to_string_lossy().into_owned();
        ctx.send_viewport_cmd(egui::ViewportCommand::Title(format!("{name} — vectorizeo")));
        let session = Session::new(l.loaded, l.class);
        self.view = View::Input;
        self.paused = false;
        self.changed_at = None;
        self.render.cancel();
        self.previews.cancel();
        self.preview = None;
        self.pending_preview = None;
        self.confirm = None;
        self.edit.color = None;
        self.doc = Some(Document {
            path: l.path,
            original,
            input_tex,
            last_settings: session.settings.clone(),
            session,
            seg_tex: None,
            result_tex: None,
            pending_render: None,
            removed: None,
        });
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn open_dialog(&mut self, _ctx: &egui::Context) {
        let mut dialog = rfd::FileDialog::new().add_filter("Images", &IMAGE_EXTENSIONS);
        if let Some(dir) = self.doc.as_ref().and_then(|d| d.path.parent()) {
            dialog = dialog.set_directory(dir);
        }
        if let Some(path) = dialog.pick_file() {
            self.open(Source { path, bytes: None });
        }
    }

    /// The browser's file picker; the file arrives through `picked`.
    #[cfg(target_arch = "wasm32")]
    fn open_dialog(&mut self, ctx: &egui::Context) {
        let dialog = rfd::AsyncFileDialog::new().add_filter("Images", &IMAGE_EXTENSIONS);
        let (tx, ctx) = (self.picked.0.clone(), ctx.clone());
        wasm_bindgen_futures::spawn_local(async move {
            if let Some(file) = dialog.pick_file().await {
                let bytes: Arc<[u8]> = file.read().await.into();
                let _ = tx.send(Source { path: PathBuf::from(file.file_name()), bytes: Some(bytes) });
                ctx.request_repaint();
            }
        });
    }

    /// Split view: the input left of the divider. Comparing from the input view shows the result.
    fn set_compare(&mut self, on: bool) {
        self.compare = on;
        if on && self.view == View::Input {
            self.view = View::Result;
        }
    }

    fn cancel(&mut self) {
        self.worker.cancel();
        self.running = None;
        self.paused = true;
        self.update_requested = false;
        self.export_pending = false;
    }

    /// Save the result in the format picked next to the Export button; waits for
    /// the result if it is not ready.
    fn export(&mut self) {
        let Some(doc) = &self.doc else { return };
        if doc.session.result().is_none() {
            self.export_pending = true;
            self.paused = false;
            self.update_requested = true;
            return;
        }
        self.export_pending = false;
        // Default: the input's name with the format's extension, next to the input.
        let format = self.export_format;
        let stem = doc.path.file_stem().unwrap_or_default().to_string_lossy();
        // The browser downloads the file.
        #[cfg(target_arch = "wasm32")]
        {
            let name = format!("{stem}.{}", format.extension());
            let text = doc.session.export(format).expect("result is current");
            let dialog = rfd::AsyncFileDialog::new().set_file_name(&name);
            wasm_bindgen_futures::spawn_local(async move {
                if let Some(file) = dialog.save_file().await {
                    let _ = file.write(text.as_bytes()).await;
                }
            });
            self.toast = Some((format!("Downloaded {name}"), true, Instant::now()));
        }
        #[cfg(not(target_arch = "wasm32"))]
        self.save_native(format, &stem.into_owned());
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn save_native(&mut self, format: Format, stem: &str) {
        let Some(doc) = &self.doc else { return };
        let mut dialog = rfd::FileDialog::new()
            .add_filter(format_name(format), &[format.extension()])
            .set_file_name(format!("{stem}.{}", format.extension()));
        if let Some(dir) = doc.path.parent() {
            dialog = dialog.set_directory(dir);
        }
        let Some(mut path) = dialog.save_file() else { return };
        // The file gets the picked format's extension.
        if !path.extension().is_some_and(|e| e.eq_ignore_ascii_case(format.extension())) {
            path.set_extension(format.extension());
        }
        let doc_text = doc.session.export(format).expect("result is current");
        let toast = match std::fs::write(&path, doc_text) {
            Ok(()) => (format!("Saved {}", path.display()), true),
            Err(e) => (format!("Cannot write {}: {e}", path.display()), false),
        };
        self.toast = Some((toast.0, toast.1, Instant::now()));
    }

    /// The last save's outcome above the status bar, for a few seconds.
    fn show_toast(&mut self, ctx: &egui::Context) {
        let Some((msg, ok, at)) = &self.toast else { return };
        let Some(left) = TOAST.checked_sub(at.elapsed()) else {
            self.toast = None;
            return;
        };
        ctx.request_repaint_after(left);
        let t = theme::tokens(ctx);
        egui::Area::new(egui::Id::new("toast"))
            .anchor(egui::Align2::CENTER_BOTTOM, vec2(0.0, -44.0))
            .interactable(false)
            .show(ctx, |ui| {
                egui::Frame::new()
                    .fill(t.panel)
                    .stroke(Stroke::new(1.0, if *ok { t.accent } else { t.danger }))
                    .corner_radius(CornerRadius::same(16))
                    .inner_margin(Margin::symmetric(14, 7))
                    .show(ui, |ui| {
                        ui.horizontal(|ui| {
                            let (dot, _) = ui.allocate_exact_size(vec2(8.0, 8.0), egui::Sense::hover());
                            ui.painter().circle_filled(dot.center(), 4.0, if *ok { t.accent } else { t.danger });
                            ui.label(msg.as_str());
                        });
                    });
            });
    }

    fn handle_input(&mut self, ctx: &egui::Context) {
        if let Some(f) = ctx.input(|i| i.raw.dropped_files.first().cloned()) {
            #[cfg(not(target_arch = "wasm32"))]
            self.open(Source { path: f.path().to_owned(), bytes: None });
            // The browser reads a dropped file asynchronously; it arrives through `picked`.
            #[cfg(target_arch = "wasm32")]
            {
                let (tx, ctx) = (self.picked.0.clone(), ctx.clone());
                wasm_bindgen_futures::spawn_local(async move {
                    if let Ok(bytes) = f.bytes_async().await {
                        let _ = tx.send(Source { path: f.path().to_owned(), bytes: Some(bytes.into()) });
                        ctx.request_repaint();
                    }
                });
            }
        }
        if let Ok(src) = self.picked.1.try_recv() {
            self.open(src);
        }
        if ctx.input_mut(|i| i.consume_shortcut(&OPEN)) {
            self.open_dialog(ctx);
        }
        if ctx.input_mut(|i| i.consume_shortcut(&SAVE)) {
            self.export();
        }
        if ctx.input(|i| i.key_pressed(Key::Escape)) && self.worker.is_busy() {
            self.cancel();
        }
        if ctx.egui_wants_keyboard_input() || self.confirm.is_some() {
            return;
        }
        let Some(doc) = &mut self.doc else { return };
        if ctx.input(|i| i.modifiers.is_none() && i.key_pressed(Key::E)) {
            self.edit.on = !self.edit.on;
        }
        let (used, changed) = self.edit.keys(ctx, self.view, &mut doc.session);
        if changed {
            self.changed_at = Some(Instant::now());
            self.paused = false;
        }
        ctx.input(|i| {
            if i.key_pressed(Key::F) && !used {
                self.canvas.fit();
            }
            if i.key_pressed(Key::Num1) {
                self.canvas.actual_size(ctx);
            }
            if i.key_pressed(Key::C) && !used {
                self.set_compare(!self.compare);
            }
            for (key, view) in [(Key::I, View::Input), (Key::S, View::Segments), (Key::R, View::Result)] {
                if i.key_pressed(key) {
                    self.view = view;
                }
            }
        });
    }

    /// Take finished work from the workers.
    fn poll(&mut self, ctx: &egui::Context) {
        if let Some(out) = self.worker.poll() {
            self.running = None;
            match out {
                Output::Loaded(Ok(l)) => self.set_document(ctx, l),
                Output::Loaded(Err(e)) | Output::Failed(e) => self.error = Some(e),
                Output::Done(done) => {
                    if let Some(doc) = &mut self.doc {
                        doc.session.accept(done);
                    }
                }
                Output::Rendered(_) | Output::Preview(_) => {}
            }
        }
        // A result belongs to the pending key (starting another task drops the previous one's).
        if let Some(out) = self.render.poll()
            && let Some(doc) = &mut self.doc
            && let Some(key) = doc.pending_render.take()
        {
            match out {
                Output::Rendered(Some(img)) => {
                    let tex = ctx.load_texture("result", image_to_color(&img), TextureOptions::LINEAR);
                    doc.result_tex = Some((key, tex));
                }
                Output::Failed(e) => self.error = Some(e),
                _ => {}
            }
        }
        if let Some(out) = self.previews.poll()
            && let Some(key) = self.pending_preview.take()
        {
            match out {
                Output::Preview(img) => {
                    self.preview = Some((key, ctx.load_texture("preview", image_to_color(&img), TEXTURE_OPTIONS)));
                }
                Output::Failed(e) => self.error = Some(e),
                _ => {}
            }
        }
        let Some(doc) = &mut self.doc else { return };
        self.edit.sync_color(&doc.session);
        if let Some(seg) = doc.session.segmentation().cloned() {
            let editor = doc.session.seg_editor();
            let rev = editor.map(|e| e.revision());
            if !doc.seg_tex.as_ref().is_some_and(|(s, r, _)| Arc::ptr_eq(s, &seg) && *r == rev) {
                let img = match editor {
                    Some(e) => ColorImage::from_rgba_unmultiplied([e.width, e.height], e.image().as_flattened()),
                    None => to_color_image(&segmentation_image(&seg.seg)),
                };
                match &mut doc.seg_tex {
                    Some((s, r, tex)) if Arc::ptr_eq(s, &seg) => {
                        tex.set(img, TEXTURE_OPTIONS);
                        *r = rev;
                    }
                    _ => doc.seg_tex = Some((seg, rev, ctx.load_texture("segments", img, TEXTURE_OPTIONS))),
                }
            }
        }
        if let Some(r) = doc.session.result() {
            let rev = doc.session.background().revision();
            if !doc.removed.as_ref().is_some_and(|(k, v, _)| Arc::ptr_eq(k, r) && *v == rev) {
                let b = doc.session.background();
                let lines = (0..r.v.colors.len())
                    .filter(|&s| b.is_hidden(s))
                    .flat_map(|s| outline(&r.v.graph, r.v.curves.as_ref(), s, OUTLINE_STEPS))
                    .collect();
                doc.removed = Some((r.clone(), rev, lines));
            }
        }
    }

    /// Start the first stale stage, or cancel a running one the settings no longer need.
    fn schedule(&mut self, ctx: &egui::Context) {
        let Some(doc) = &mut self.doc else { return };
        if self.confirm.is_some() {
            return;
        }
        if doc.session.settings != doc.last_settings {
            // A change that re-segments would discard the manual edits: ask first.
            if doc.session.settings_discard_edits() {
                self.confirm = Some(doc.last_settings.clone());
                return;
            }
            doc.last_settings = doc.session.settings.clone();
            self.changed_at = Some(Instant::now());
            self.paused = false;
        }
        // The palettes are listed in the settings panel even when the choice does not need them.
        let next = doc.session.next_job(Target::Palettes).or_else(|| doc.session.next_job(Target::Result));
        if let Some(running) = &self.running {
            if !next.as_ref().is_some_and(|n| n.same_as(running)) {
                self.worker.cancel();
                self.running = None;
            }
            return;
        }
        if self.worker.is_busy() || self.paused || !(self.auto_update || self.update_requested) {
            return;
        }
        let Some(job) = next else {
            self.update_requested = false;
            return;
        };
        if let Some(wait) = self.changed_at.and_then(|t| DEBOUNCE.checked_sub(t.elapsed())) {
            ctx.request_repaint_after(wait);
            return;
        }
        self.running = Some(job.clone());
        self.worker.spawn(Task::Stage(job));
    }

    /// Rasterise the current result at about the canvas zoom (in √2 steps).
    fn schedule_render(&mut self, ctx: &egui::Context) {
        let Some(doc) = &mut self.doc else { return };
        let side = doc.original.width().max(doc.original.height()) as f32;
        let want = (self.canvas.zoom() * ctx.pixels_per_point()).max(0.05);
        let scale = 2f32.powf((want.log2() * 2.0).ceil() / 2.0).min(MAX_RENDER / side);
        let Some(key) = doc.render_key(scale) else { return };
        if doc.result_tex.as_ref().is_some_and(|(k, _)| *k == key) || doc.pending_render.as_ref() == Some(&key) {
            return;
        }
        // The export walk is cheap; the worker only rasterises.
        let v = &key.result.v;
        let svg = export_svg_hidden(&v.graph, v.curves.as_ref(), &v.colors, key.export, &key.hidden);
        self.render.spawn(Task::Render { svg, scale: key.scale });
        doc.pending_render = Some(key);
    }

    /// Compute the preview of the hovered option.
    fn schedule_preview(&mut self) {
        let (Some(doc), Some(hover)) = (&self.doc, &self.hover) else { return };
        let key = (hover.clone(), doc.session.settings.flatten);
        if self.preview.as_ref().is_some_and(|(k, _)| *k == key) || self.pending_preview.as_ref() == Some(&key) {
            return;
        }
        let (flatten, palette) = match &key {
            (Hover::Flatten(f), _) => (*f, None),
            (Hover::Palette(p), f) => (*f, Some(p.clone())),
        };
        self.previews.spawn(Task::Preview { loaded: doc.session.loaded.clone(), flatten, palette });
        self.pending_preview = Some(key);
    }

    fn top_bar(&mut self, ui: &mut Ui) {
        let t = theme::tokens(ui.ctx());
        egui::Panel::top("top").frame(egui::Frame::new().fill(t.panel).inner_margin(Margin::symmetric(12, 8))).show(
            ui,
            |ui| {
                ui.horizontal(|ui| {
                    theme::logo(ui);
                    ui.label(RichText::new("vectorizeo").strong().size(15.0));
                    ui.add_space(8.0);
                    if ui.button("Open…").on_hover_text("Open an image (Ctrl+O)").clicked() {
                        self.open_dialog(ui.ctx());
                    }
                    if let Some(doc) = &self.doc {
                        let name = doc.path.file_name().unwrap_or_default().to_string_lossy();
                        ui.label(RichText::new(name).color(t.muted));
                    }
                    ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                        let label = match (self.export_pending, self.export_format) {
                            (true, _) => "Finishing…",
                            (false, Format::Svg) => "Export SVG",
                            (false, Format::Eps) => "Export EPS",
                        };
                        let current = self.export_format;
                        let export = ui.add_enabled_ui(self.doc.is_some() && !self.export_pending, |ui| {
                            accent_split_button(ui, label, |ui| {
                                let mut picked = None;
                                for f in [Format::Svg, Format::Eps] {
                                    if ui.add(egui::Button::selectable(f == current, format_name(f))).clicked() {
                                        picked = Some(f);
                                    }
                                }
                                picked
                            })
                        });
                        let (main, picked) = export.inner;
                        // Picking a format exports in it right away.
                        if let Some(f) = picked.flatten() {
                            self.export_format = f;
                            self.export();
                        } else if main.on_hover_text("Save the result (Ctrl+S); ⏷ picks SVG or EPS").clicked() {
                            self.export();
                        }
                        theme_switch(ui);
                        ui.separator();
                        ui.add_enabled_ui(self.doc.is_some(), |ui| {
                            ui.toggle_value(&mut self.edit.on, "Edit").on_hover_text(
                                "Edit the segmentation (Segments view) or remove background shapes (Result view) (E)",
                            );
                            let mut compare = self.compare;
                            if ui
                                .toggle_value(&mut compare, "Compare")
                                .on_hover_text("Input on the left, the view on the right (C)")
                                .changed()
                            {
                                self.set_compare(compare);
                            }
                            segmented(
                                ui,
                                &mut self.view,
                                &[(View::Input, "Input"), (View::Segments, "Segments"), (View::Result, "Result")],
                            )
                            .on_hover_text("Views: I, S, R");
                        });
                    });
                });
            },
        );
    }

    fn side_panel(&mut self, ui: &mut Ui) {
        let t = theme::tokens(ui.ctx());
        self.hover = None;
        egui::Panel::right("settings")
            .default_size(330.0)
            .min_size(280.0)
            .frame(egui::Frame::new().fill(t.panel).inner_margin(Margin::same(14)))
            .show(ui, |ui| {
                egui::ScrollArea::vertical().show(ui, |ui| {
                    ui.set_width(ui.available_width());
                    caption(ui, "Image");
                    let Some(doc) = &mut self.doc else {
                        ui.label(RichText::new("No image open").color(t.muted));
                        return;
                    };
                    let (w, h) = doc.original.dimensions();
                    let alpha = if doc.session.has_alpha { " · transparency" } else { "" };
                    ui.label(format!("{w} × {h} px{alpha}"));
                    let class = doc.session.class.clone();
                    ui.label(
                        RichText::new(format!(
                            "Detected: {}, {} quality",
                            type_name(class.image_type),
                            quality_name(class.quality).to_lowercase()
                        ))
                        .color(t.muted),
                    )
                    .on_hover_text(format!("Classifier leaf \"{}\"", class.label));
                    ui.add_space(4.0);
                    let palettes = doc.session.palettes().cloned();
                    let inputs = inspector::Inputs {
                        class: &class,
                        has_alpha: doc.session.has_alpha,
                        palettes: palettes.as_deref(),
                    };
                    self.hover = inspector::show(ui, &mut doc.session.settings, &inputs);
                });
            });
    }

    fn status_bar(&mut self, ui: &mut Ui) {
        let t = theme::tokens(ui.ctx());
        egui::Panel::bottom("status")
            .frame(egui::Frame::new().fill(t.panel).inner_margin(Margin::symmetric(12, 4)))
            .show(ui, |ui| {
                ui.horizontal(|ui| {
                    self.progress(ui);
                    ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                        if self.doc.is_none() {
                            return;
                        }
                        ui.spacing_mut().item_spacing.x = 4.0;
                        if ui.small_button("Fit").on_hover_text("Fit to window (F, double-click)").clicked() {
                            self.canvas.fit();
                        }
                        if ui.small_button("1:1").on_hover_text("Actual pixels (1)").clicked() {
                            self.canvas.actual_size(ui.ctx());
                        }
                        if ui.small_button("+").clicked() {
                            self.canvas.zoom_by(1.25);
                        }
                        let zoom = self.canvas.zoom() * ui.ctx().pixels_per_point();
                        ui.label(RichText::new(format!("{:.0}%", zoom * 100.0)).monospace());
                        if ui.small_button("−").clicked() {
                            self.canvas.zoom_by(0.8);
                        }
                        if let Some([x, y]) = self.canvas.hover_px {
                            ui.add_space(12.0);
                            ui.label(RichText::new(format!("{x}, {y}")).monospace().color(t.muted));
                        }
                        ui.add_space(12.0);
                        ui.checkbox(&mut self.auto_update, "Auto update")
                            .on_hover_text("Re-run the affected steps whenever a setting changes");
                    });
                });
            });
    }

    /// Left side of the status bar: the running stage with Cancel, or what is left to do.
    fn progress(&mut self, ui: &mut Ui) {
        let t = theme::tokens(ui.ctx());
        if let Some(busy) = self.worker.busy() {
            ui.add(egui::Spinner::new().size(14.0).color(t.accent));
            ui.label(format!("{}…", busy.label));
            let secs = busy.started.elapsed().as_secs_f32();
            ui.label(RichText::new(format!("{secs:.1} s")).monospace().color(t.muted));
            if ui.small_button("Cancel").on_hover_text("Stop this step (Esc)").clicked() {
                self.cancel();
            }
            ui.ctx().request_repaint_after(Duration::from_millis(100));
            return;
        }
        if let Some(e) = &self.error {
            ui.label(RichText::new(e).color(t.danger));
            return;
        }
        let Some(doc) = &self.doc else { return };
        if doc.session.next_job(Target::Result).is_some() {
            let why = if self.paused { "Cancelled" } else { "Settings changed" };
            ui.label(RichText::new(why).color(t.muted));
            if (self.paused || !self.auto_update) && accent_button(ui, "Update").clicked() {
                self.paused = false;
                self.update_requested = true;
            }
        } else if let Some(r) = doc.session.result() {
            let segments = doc.session.segmentation().map_or(0, |s| s.seg.segments.len());
            let curves = r.v.curves.as_ref().map_or(0, |c| c.fits.len());
            ui.label(RichText::new(format!("{segments} segments · {curves} curves")).color(t.muted));
        }
    }

    fn central(&mut self, ui: &mut Ui) {
        let t = theme::tokens(ui.ctx());
        egui::CentralPanel::default().frame(egui::Frame::new().fill(t.canvas)).show(ui, |ui| {
            let Some(doc) = &self.doc else {
                self.empty_state(ui);
                return;
            };
            let input = Layer { tex: doc.input_tex.id(), label: "Input", dim: false };
            let (main, fresh) = match self.view {
                View::Input => (input, true),
                View::Segments => match &doc.seg_tex {
                    Some((_, _, tex)) => {
                        (Layer { tex: tex.id(), label: "Segments", dim: !doc.seg_fresh() }, doc.seg_fresh())
                    }
                    None => (Layer { dim: true, ..input }, false),
                },
                View::Result => match &doc.result_tex {
                    Some((_, tex)) => {
                        (Layer { tex: tex.id(), label: "Result", dim: !doc.result_fresh() }, doc.result_fresh())
                    }
                    None => (Layer { dim: true, ..input }, false),
                },
            };
            let before = (self.compare && self.view != View::Input).then_some(input);
            let rect = ui.max_rect();
            // A hovered option in the settings panel is previewed in place of the view.
            let preview = self.hover.as_ref().and_then(|h| {
                let (key, tex) = self.preview.as_ref()?;
                (key.0 == *h && key.1 == doc.session.settings.flatten).then_some(tex)
            });
            if let Some(tex) = preview {
                self.canvas.show(
                    ui,
                    Some(Layer { tex: tex.id(), label: "Preview", dim: false }),
                    Some(input).filter(|_| self.compare),
                    false,
                );
                return;
            }
            // Edits keep the view current: only a new segmentation or result makes it stale.
            let ready = match self.view {
                View::Segments => doc.session.segmentation().is_some(),
                View::Result => doc.session.result().is_some(),
                View::Input => false,
            };
            let tool = self.edit.active(self.view).is_some() && ready;
            let pointer = self.canvas.show(ui, Some(main), before, tool);
            let Some(doc) = &mut self.doc else { return };
            if self.view == View::Result
                && ready
                && let Some((r, _, lines)) = &doc.removed
                && doc.session.result().is_some_and(|cur| Arc::ptr_eq(cur, r))
            {
                crate::edit::draw_removed(ui, rect, &self.canvas, lines);
            }
            let mut changed = tool && self.edit.pointer(self.view, pointer, &mut doc.session);
            changed |= self.edit.toolbar(ui, rect, self.view, &mut doc.session);
            if changed {
                self.changed_at = Some(Instant::now());
                self.paused = false;
            }
            if !fresh && let Some(busy) = self.worker.busy() {
                working_pill(ui, rect, &busy.label);
            }
        });
    }

    /// Drop zone shown before an image is open.
    fn empty_state(&mut self, ui: &mut Ui) {
        let t = theme::tokens(ui.ctx());
        let hovering = ui.ctx().input(|i| !i.raw.hovered_files.is_empty());
        let zone = egui::Rect::from_center_size(ui.max_rect().center(), vec2(420.0, 240.0));
        let stroke = Stroke::new(1.5, if hovering { t.accent } else { t.border });
        let fill = if hovering { t.accent_soft } else { t.panel };
        ui.painter().rect_filled(zone, CornerRadius::ZERO, fill);
        let pts = [zone.left_top(), zone.right_top(), zone.right_bottom(), zone.left_bottom(), zone.left_top()];
        ui.painter().extend(egui::Shape::dashed_line(&pts, stroke, 6.0, 5.0));
        ui.scope_builder(egui::UiBuilder::new().max_rect(zone.shrink(24.0)), |ui| {
            ui.vertical_centered(|ui| {
                ui.add_space(28.0);
                ui.label(RichText::new("⬆").size(30.0).color(if hovering { t.accent } else { t.muted }));
                ui.label(RichText::new("Drop an image here").size(17.0).strong());
                ui.label(RichText::new("PNG, JPEG, BMP, GIF or TIFF").color(t.muted));
                ui.add_space(10.0);
                if let Some(busy) = self.worker.busy() {
                    ui.horizontal(|ui| {
                        ui.add(egui::Spinner::new().color(t.accent));
                        ui.label(format!("{}…", busy.label));
                    });
                } else if accent_button(ui, "Open image…").clicked() {
                    self.open_dialog(ui.ctx());
                }
            });
        });
    }
}

fn format_name(f: Format) -> &'static str {
    match f {
        Format::Svg => "SVG (Scalable Vector Graphics)",
        Format::Eps => "EPS (Encapsulated PostScript)",
    }
}

impl App {
    /// "You have made changes to the segmentation…" for a settings change that
    /// re-segments: overwrite the edits, or go back to the previous settings.
    fn confirm_dialog(&mut self, ctx: &egui::Context) {
        let (Some(prev), Some(doc)) = (&self.confirm, &mut self.doc) else { return };
        let mut answer = None;
        egui::Modal::new(egui::Id::new("discard_edits")).show(ctx, |ui| {
            ui.set_max_width(360.0);
            ui.label(RichText::new("Overwrite segmentation edits?").strong().size(15.0));
            ui.add_space(4.0);
            ui.label(
                "You have made changes to the segmentation. This setting re-segments the image, which discards them.",
            );
            ui.add_space(8.0);
            ui.horizontal(|ui| {
                if accent_button(ui, "Overwrite").clicked() {
                    answer = Some(true);
                }
                if ui.button("Keep edits").clicked() || ui.input(|i| i.key_pressed(Key::Escape)) {
                    answer = Some(false);
                }
            });
        });
        match answer {
            Some(true) => doc.session.discard_seg_edits(),
            Some(false) => doc.session.settings = prev.clone(),
            None => return,
        }
        self.confirm = None;
    }
}

/// "Segmenting Image…" over the top of the canvas while the shown view is computed.
fn working_pill(ui: &mut Ui, canvas: egui::Rect, label: &str) {
    let t = theme::tokens(ui.ctx());
    let pill = egui::Rect::from_center_size(pos2(canvas.center().x, canvas.top() + 30.0), vec2(240.0, 32.0));
    ui.scope_builder(egui::UiBuilder::new().max_rect(pill), |ui| {
        egui::Frame::new()
            .fill(t.panel)
            .stroke(Stroke::new(1.0, t.border))
            .corner_radius(CornerRadius::same(16))
            .inner_margin(Margin::symmetric(12, 6))
            .show(ui, |ui| {
                ui.horizontal(|ui| {
                    ui.add(egui::Spinner::new().size(14.0).color(t.accent));
                    ui.label(format!("{label}…"));
                });
            });
    });
}

/// System / light / dark.
fn theme_switch(ui: &mut Ui) {
    let mut pref = ui.ctx().options(|o| o.theme_preference);
    let before = pref;
    segmented(
        ui,
        &mut pref,
        &[(ThemePreference::System, "Auto"), (ThemePreference::Light, "☀"), (ThemePreference::Dark, "🌙")],
    )
    .on_hover_text("Theme");
    if pref != before {
        ui.ctx().set_theme(pref);
    }
}

impl eframe::App for App {
    fn ui(&mut self, ui: &mut Ui, _frame: &mut eframe::Frame) {
        let ctx = ui.ctx().clone();
        self.poll(&ctx);
        self.handle_input(&ctx);
        self.top_bar(ui);
        self.status_bar(ui);
        self.side_panel(ui);
        self.central(ui);
        self.schedule(&ctx);
        self.schedule_render(&ctx);
        self.schedule_preview();
        if self.export_pending && self.doc.as_ref().is_some_and(|d| d.session.svg().is_some()) {
            self.export();
        }
        self.show_toast(&ctx);
        self.confirm_dialog(&ctx);
    }

    fn clear_color(&self, _visuals: &egui::Visuals) -> [f32; 4] {
        Color32::TRANSPARENT.to_normalized_gamma_f32()
    }
}
