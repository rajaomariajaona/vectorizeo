//! The image canvas: zoom (wheel / pinch at the cursor), pan (drag), fit, a checkerboard under
//! transparency, and a before/after split with a draggable divider. With an edit tool active the
//! primary button goes to the tool instead (Shift + drag or the middle button still pans).

use eframe::egui::{
    self, Align2, Color32, CornerRadius, CursorIcon, FontId, Pos2, Rect, Sense, Shape, Stroke, TextureId, Ui, Vec2,
    pos2, vec2,
};

use crate::theme::{self, Tokens};

const MIN_ZOOM: f32 = 0.05;
const MAX_ZOOM: f32 = 64.0;

/// The pointer in image pixel coordinates: the primary button for an edit tool, or hover and
/// clicks without one.
#[derive(Clone, Copy, Debug, Default)]
pub struct Pointer {
    /// Pointer position (may lie outside the image); without a tool only while over the canvas.
    pub pos: Option<[f64; 2]>,
    /// Clicked without dragging (no tool active).
    pub clicked: bool,
    /// Pressed on the canvas this frame.
    pub pressed: bool,
    /// Held since a press on the canvas.
    pub down: bool,
    /// Released after a press on the canvas.
    pub released: bool,
}

/// One image drawn over the whole image rectangle (textures of any resolution).
#[derive(Clone, Copy)]
pub struct Layer {
    pub tex: TextureId,
    pub label: &'static str,
    /// Drawn faded: an older result shown while the current one is computed.
    pub dim: bool,
}

pub struct Canvas {
    /// Image size in pixels.
    size: Vec2,
    /// Screen pixels per image pixel.
    zoom: f32,
    /// Offset of the image centre from the canvas centre.
    pan: Vec2,
    /// Keep the image fitted to the canvas until the user zooms or pans.
    fitting: bool,
    /// Divider position as a fraction of the canvas width.
    split: f32,
    /// Image pixel under the pointer.
    pub hover_px: Option<[i32; 2]>,
    /// Screen rectangle of the image in the last frame.
    image_rect: Rect,
    /// The primary button went down on the canvas with a tool active.
    tool_drag: bool,
}

impl Default for Canvas {
    fn default() -> Self {
        Self {
            size: Vec2::splat(1.0),
            zoom: 1.0,
            pan: Vec2::ZERO,
            fitting: true,
            split: 0.5,
            hover_px: None,
            image_rect: Rect::NOTHING,
            tool_drag: false,
        }
    }
}

impl Canvas {
    pub fn reset(&mut self, width: usize, height: usize) {
        *self = Self { size: vec2(width as f32, height as f32), ..Self::default() };
    }

    pub fn zoom(&self) -> f32 {
        self.zoom
    }

    pub fn fit(&mut self) {
        self.fitting = true;
    }

    /// One image pixel per screen pixel (physical pixels).
    pub fn actual_size(&mut self, ctx: &egui::Context) {
        self.fitting = false;
        self.zoom = 1.0 / ctx.pixels_per_point();
        self.pan = Vec2::ZERO;
    }

    /// Zoom about the canvas centre.
    pub fn zoom_by(&mut self, factor: f32) {
        self.fitting = false;
        let z = (self.zoom * factor).clamp(MIN_ZOOM, MAX_ZOOM);
        self.pan *= z / self.zoom;
        self.zoom = z;
    }

    /// Screen position of an image point (last frame's layout).
    pub fn to_screen(&self, p: [f64; 2]) -> Pos2 {
        self.image_rect.min + vec2(p[0] as f32, p[1] as f32) * self.zoom
    }

    /// Draw `main`; with `before`, the part left of the divider shows `before` instead. With
    /// `tool`, primary-button input is returned for the edit tool instead of panning.
    pub fn show(&mut self, ui: &mut Ui, main: Option<Layer>, before: Option<Layer>, tool: bool) -> Pointer {
        let t = theme::tokens(ui.ctx());
        let rect = ui.available_rect_before_wrap();
        let response = ui.allocate_rect(rect, Sense::click_and_drag());
        let painter = ui.painter_at(rect);
        painter.rect_filled(rect, CornerRadius::ZERO, t.canvas);

        if self.fitting {
            let avail = (rect.size() - Vec2::splat(48.0)).max(Vec2::splat(16.0));
            self.zoom = (avail.x / self.size.x).min(avail.y / self.size.y).clamp(MIN_ZOOM, MAX_ZOOM);
            self.pan = Vec2::ZERO;
        }

        // Wheel and pinch zoom about the pointer; drag pans.
        if let Some(cursor) = response.hover_pos() {
            let (scroll, pinch) = ui.input(|i| (i.smooth_scroll_delta.y, i.zoom_delta()));
            let factor = pinch * (scroll * 0.0015).exp();
            if factor != 1.0 {
                self.fitting = false;
                let q = (cursor - rect.center() - self.pan) / self.zoom;
                self.zoom = (self.zoom * factor).clamp(MIN_ZOOM, MAX_ZOOM);
                self.pan = cursor - rect.center() - q * self.zoom;
            }
        }
        let (shift, primary_pressed, primary_down, latest) = ui.input(|i| {
            (i.modifiers.shift, i.pointer.primary_pressed(), i.pointer.primary_down(), i.pointer.latest_pos())
        });
        // Shift switches to panning for a moment.
        let tool = tool && !shift;
        if response.double_clicked() && !tool {
            self.fit();
        }

        let image = Rect::from_center_size(rect.center() + self.pan, self.size * self.zoom);
        self.image_rect = image;
        let to_image = |p: Pos2| {
            let q = (p - image.min) / self.zoom;
            [f64::from(q.x), f64::from(q.y)]
        };
        let pointer = if tool {
            let pressed = primary_pressed && response.hovered();
            if pressed {
                self.tool_drag = true;
            }
            let pointer = Pointer {
                pos: latest.map(to_image),
                pressed,
                down: self.tool_drag && primary_down,
                released: self.tool_drag && !primary_down,
                clicked: false,
            };
            if !primary_down {
                self.tool_drag = false;
            }
            if response.hovered() {
                ui.ctx().set_cursor_icon(CursorIcon::Crosshair);
            }
            pointer
        } else {
            self.tool_drag = false;
            Pointer {
                pos: response.hover_pos().map(to_image),
                clicked: response.clicked() && !shift,
                ..Pointer::default()
            }
        };
        let split_x = rect.left() + self.split * rect.width();
        let handle = before.map(|_| {
            let r = Rect::from_x_y_ranges(split_x - 8.0..=split_x + 8.0, rect.y_range());
            ui.interact(r, ui.id().with("split"), Sense::drag()).on_hover_cursor(CursorIcon::ResizeHorizontal)
        });
        // The middle button always pans, also with an edit tool active.
        let panning = response.dragged_by(egui::PointerButton::Middle) || (response.dragged() && !tool);
        match &handle {
            Some(h) if h.dragged() => {
                if let Some(p) = h.interact_pointer_pos() {
                    self.split = ((p.x - rect.left()) / rect.width()).clamp(0.0, 1.0);
                }
            }
            _ if panning => {
                self.fitting = false;
                self.pan += response.drag_delta();
            }
            _ => {}
        }
        if panning {
            ui.ctx().set_cursor_icon(CursorIcon::Grabbing);
        }

        self.hover_px = response.hover_pos().and_then(|p| {
            let q = (p - image.min) / self.zoom;
            (q.x >= 0.0 && q.y >= 0.0 && q.x < self.size.x && q.y < self.size.y).then(|| [q.x as i32, q.y as i32])
        });

        checkerboard(&painter, image, &t);
        let uv = Rect::from_min_max(Pos2::ZERO, pos2(1.0, 1.0));
        let draw = |clip: Rect, layer: Layer| {
            let tint = if layer.dim { Color32::from_white_alpha(110) } else { Color32::WHITE };
            painter.with_clip_rect(clip.intersect(rect)).image(layer.tex, image, uv, tint);
        };
        match (main, before) {
            (Some(main), Some(before)) => {
                let (l, r) = rect.split_left_right_at_x(split_x);
                draw(l, before);
                draw(r, main);
                self.divider(&painter, rect, split_x, handle.is_some_and(|h| h.hovered() || h.dragged()), &t);
                chip(&painter, rect.left_top() + vec2(12.0, 12.0), Align2::LEFT_TOP, before.label, &t);
                chip(&painter, rect.right_top() + vec2(-12.0, 12.0), Align2::RIGHT_TOP, main.label, &t);
            }
            (Some(main), None) => draw(rect, main),
            (None, _) => {}
        }
        painter.rect_stroke(image, CornerRadius::ZERO, Stroke::new(1.0, t.border), egui::StrokeKind::Outside);
        pointer
    }

    fn divider(&self, painter: &egui::Painter, rect: Rect, x: f32, hot: bool, t: &Tokens) {
        let color = if hot { t.accent } else { Color32::WHITE };
        // A faint shadow keeps the white line visible over white images.
        let line = [pos2(x, rect.top()), pos2(x, rect.bottom())];
        painter.line_segment(line, Stroke::new(4.0, Color32::from_black_alpha(50)));
        painter.line_segment(line, Stroke::new(2.0, color));
        let c = pos2(x, rect.center().y);
        painter.circle(c, 14.0, color, Stroke::new(1.0, Color32::from_black_alpha(60)));
        let ink = if hot { t.on_accent } else { Color32::from_gray(40) };
        for dir in [-1.0f32, 1.0] {
            let tip = c + vec2(dir * 8.0, 0.0);
            let base = c + vec2(dir * 3.0, 0.0);
            painter.add(Shape::convex_polygon(
                vec![tip, base + vec2(0.0, -4.5), base + vec2(0.0, 4.5)],
                ink,
                Stroke::NONE,
            ));
        }
    }
}

/// Checkerboard in screen space behind the image (only its visible part).
fn checkerboard(painter: &egui::Painter, image: Rect, t: &Tokens) {
    const CELL: f32 = 10.0;
    let vis = image.intersect(painter.clip_rect());
    if !vis.is_positive() {
        return;
    }
    painter.rect_filled(vis, CornerRadius::ZERO, t.checker[0]);
    let (i0, j0) = (((vis.left() - image.left()) / CELL) as i64, ((vis.top() - image.top()) / CELL) as i64);
    let (i1, j1) = (((vis.right() - image.left()) / CELL) as i64, ((vis.bottom() - image.top()) / CELL) as i64);
    for j in j0..=j1 {
        for i in i0..=i1 {
            if (i + j) % 2 == 1 {
                let min = image.min + vec2(i as f32, j as f32) * CELL;
                let cell = Rect::from_min_size(min, Vec2::splat(CELL)).intersect(vis);
                painter.rect_filled(cell, CornerRadius::ZERO, t.checker[1]);
            }
        }
    }
}

fn chip(painter: &egui::Painter, pos: Pos2, align: Align2, text: &str, t: &Tokens) {
    let galley = painter.layout_no_wrap(text.to_owned(), FontId::proportional(11.5), t.text);
    let rect = align.anchor_size(pos, galley.size() + vec2(16.0, 8.0));
    painter.rect(
        rect,
        CornerRadius::same(8),
        t.panel.gamma_multiply(0.92),
        Stroke::new(1.0, t.border),
        egui::StrokeKind::Inside,
    );
    painter.galley(rect.min + vec2(8.0, 4.0), galley, t.text);
}
