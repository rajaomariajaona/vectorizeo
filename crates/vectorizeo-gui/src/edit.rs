//! Editing. In the Result view a click on a shape opens a menu: remove or restore it, recolour
//! it, or merge it into its neighbours. In the Segments view, Paint pixels gives the pixel tools
//! (keys, no modifier: B pencil, G fill, M merge, K pick a colour; Shift pans for a moment). Both
//! kinds of edit share one history (Ctrl+Z undo, Ctrl+Shift+Z or Ctrl+Y redo).

use std::sync::Arc;

use eframe::egui::{
    self, Color32, CornerRadius, Key, KeyboardShortcut, Margin, Modifiers, Pos2, Rect, RichText, Shape, Stroke, Ui,
    vec2,
};
use vectorizeo_core::export::outline;
use vectorizeo_pipeline::edit::Action;
use vectorizeo_pipeline::hex_rgba;
use vectorizeo_pipeline::session::{Session, VecRun};

use crate::app::View;
use crate::canvas::{Canvas, Pointer};
use crate::theme::{self, accent_button};

/// Pieces per cubic in shape outlines.
const OUTLINE_STEPS: usize = 12;
/// Colours offered by Recolour.
const MAX_SWATCHES: usize = 24;
const UNDO: KeyboardShortcut = KeyboardShortcut::new(Modifiers::COMMAND, Key::Z);
const REDO: KeyboardShortcut = KeyboardShortcut::new(Modifiers::COMMAND.plus(Modifiers::SHIFT), Key::Z);
const REDO_Y: KeyboardShortcut = KeyboardShortcut::new(Modifiers::COMMAND, Key::Y);

/// Paint pixels tools.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Tool {
    Pan,
    Pencil,
    Fill,
    Merge,
    Picker,
}

const TOOLS: [Tool; 5] = [Tool::Pan, Tool::Pencil, Tool::Fill, Tool::Merge, Tool::Picker];

impl Tool {
    fn label(self) -> &'static str {
        match self {
            Tool::Pan => "Pan",
            Tool::Pencil => "Pencil",
            Tool::Fill => "Fill",
            Tool::Merge => "Merge",
            Tool::Picker => "Pick colour",
        }
    }

    fn key(self) -> Option<Key> {
        match self {
            Tool::Pan => None,
            Tool::Pencil => Some(Key::B),
            Tool::Fill => Some(Key::G),
            Tool::Merge => Some(Key::M),
            Tool::Picker => Some(Key::K),
        }
    }

    fn tip(self) -> &'static str {
        match self {
            Tool::Pan => "Drag to move the image",
            Tool::Pencil => "Draw with the current colour",
            Tool::Fill => "Fill the connected area of one colour with the current colour",
            Tool::Merge => {
                "Give each pixel of an area the closest neighbouring colour (anti-aliasing remnants, specks)"
            }
            Tool::Picker => "Pick the current colour from the segments, then return to the previous tool",
        }
    }
}

/// The menu of a clicked shape in the Result view.
struct ShapeMenu {
    /// The result the shape belongs to; another result closes the menu.
    result: Arc<VecRun>,
    /// The clicked image point.
    point: [f64; 2],
    shape: usize,
    /// The colour Recolour's picker holds.
    custom: Color32,
}

pub struct EditTools {
    /// Paint pixels in the Segments view.
    pub paint: bool,
    tool: Tool,
    /// The tool the colour picker returns to.
    prev_tool: Tool,
    /// The pencil and fill colour (the swatch, palette[0] after a segmentation).
    pub color: Option<[u8; 4]>,
    /// The colour under the picker.
    hover_color: Option<[u8; 4]>,
    /// Previous pencil position while the button is held.
    last: Option<[f64; 2]>,
    menu: Option<ShapeMenu>,
    /// The shape under the pointer in the Result view, with its outline.
    hovered: Option<(Arc<VecRun>, usize, Vec<Vec<[f64; 2]>>)>,
}

impl Default for EditTools {
    fn default() -> Self {
        Self {
            paint: false,
            tool: Tool::Pencil,
            prev_tool: Tool::Pencil,
            color: None,
            hover_color: None,
            last: None,
            menu: None,
            hovered: None,
        }
    }
}

impl EditTools {
    /// Forget the menu and hover of the previous image.
    pub fn reset(&mut self) {
        self.color = None;
        self.menu = None;
        self.hovered = None;
    }

    /// The paint tool the canvas hands the primary button to in `view`, if any.
    pub fn active(&self, view: View) -> Option<Tool> {
        (view == View::Segments && self.paint && self.tool != Tool::Pan).then_some(self.tool)
    }

    /// Clicking the selected tool again goes back to Pan.
    fn select(&mut self, tool: Tool) {
        if self.tool == tool && tool != Tool::Pan {
            self.tool = Tool::Pan;
        } else {
            if tool == Tool::Picker && self.tool != Tool::Picker {
                self.prev_tool = self.tool;
            }
            self.tool = tool;
        }
        self.last = None;
    }

    /// Undo/redo in every view, the paint tool keys in the Segments view, Esc closes the shape
    /// menu. Returns whether an edit changed.
    pub fn keys(&mut self, ctx: &egui::Context, view: View, session: &mut Session) -> bool {
        let mut changed = false;
        ctx.input_mut(|i| {
            if i.consume_shortcut(&REDO) || i.consume_shortcut(&REDO_Y) {
                changed |= session.redo();
            } else if i.consume_shortcut(&UNDO) {
                changed |= session.undo();
            }
            if i.key_pressed(Key::Escape) {
                self.menu = None;
            }
            if view != View::Segments || !self.paint || !i.modifiers.is_none() {
                return;
            }
            for t in TOOLS {
                if t.key().is_some_and(|k| i.key_pressed(k)) {
                    self.select(t);
                }
            }
        });
        changed
    }

    /// Apply the pointer: the paint tool in the Segments view, hover and clicks on shapes in the
    /// Result view. Returns whether an edit changed.
    pub fn pointer(&mut self, view: View, p: Pointer, session: &mut Session) -> bool {
        self.hover_color = None;
        if view == View::Result {
            self.shapes(p, session);
            return false;
        }
        let Some(tool) = self.active(view) else { return false };
        let Some(pos) = p.pos else { return false };
        let px = pos.map(|v| v as i32);
        match tool {
            Tool::Picker => {
                self.hover_color = session.color_at(px[0], px[1]);
                if p.released {
                    if let Some(c) = self.hover_color {
                        self.color = Some(c);
                    }
                    self.tool = self.prev_tool;
                }
                false
            }
            Tool::Pencil => {
                let Some(color) = self.color else { return false };
                if p.released || !p.down {
                    self.last = None;
                    return false;
                }
                // Press: from = to; every move with the button down: previous → current.
                let from = if p.pressed { pos } else { self.last.unwrap_or(pos) };
                if !p.pressed && self.last == Some(pos) {
                    return false;
                }
                self.last = Some(pos);
                session.edit_segments(Action::Pencil { from, to: pos, color })
            }
            Tool::Fill | Tool::Merge => {
                if !p.pressed {
                    return false;
                }
                let (x, y) = (px[0], px[1]);
                match (tool, self.color) {
                    (Tool::Fill, Some(color)) => session.edit_segments(Action::Fill { x, y, color }),
                    (Tool::Merge, _) => session.edit_segments(Action::Zap { x, y }),
                    _ => false,
                }
            }
            Tool::Pan => false,
        }
    }

    /// The hovered shape's outline, and a click opens its menu (or closes the menu off shapes).
    fn shapes(&mut self, p: Pointer, session: &Session) {
        let Some(r) = session.result().cloned() else {
            self.hovered = None;
            return;
        };
        let shape = p.pos.and_then(|pos| session.shape_at(pos));
        if p.clicked {
            self.menu = shape.zip(p.pos).map(|(shape, point)| {
                let c = r.v.colors[shape];
                ShapeMenu {
                    result: r.clone(),
                    point,
                    shape,
                    custom: Color32::from_rgba_unmultiplied(c[0], c[1], c[2], c[3]),
                }
            });
        }
        match shape {
            Some(s) if !self.hovered.as_ref().is_some_and(|(h, hs, _)| Arc::ptr_eq(h, &r) && *hs == s) => {
                let lines = outline(&r.v.graph, r.v.curves.as_ref(), s, OUTLINE_STEPS);
                self.hovered = Some((r, s, lines));
            }
            Some(_) => {}
            None => self.hovered = None,
        }
    }

    /// Everything drawn over the canvas: the hovered shape, the shape menu and the strip with
    /// the history buttons (and the paint tools in the Segments view). Returns whether an edit
    /// changed.
    pub fn overlay(
        &mut self,
        ui: &mut Ui,
        canvas_rect: Rect,
        view: View,
        canvas: &Canvas,
        session: &mut Session,
    ) -> bool {
        let mut changed = false;
        let result = session.result().cloned();
        if view != View::Result
            || !self.menu.as_ref().is_some_and(|m| result.as_ref().is_some_and(|r| Arc::ptr_eq(r, &m.result)))
        {
            self.menu = None;
        }
        if view == View::Result {
            let t = theme::tokens(ui.ctx());
            // The menu's shape stays outlined while the menu is open.
            let lines = match (&self.menu, &self.hovered) {
                (Some(m), _) => Some(outline(&m.result.v.graph, m.result.v.curves.as_ref(), m.shape, OUTLINE_STEPS)),
                (None, Some((r, _, lines))) if result.as_ref().is_some_and(|c| Arc::ptr_eq(c, r)) => {
                    Some(lines.clone())
                }
                _ => None,
            };
            if let Some(lines) = lines {
                draw_outline(ui, canvas_rect, canvas, &lines, t.accent);
            }
            changed |= self.shape_menu(ui, canvas, session);
        }
        let ready = match view {
            View::Segments => session.segmentation().is_some(),
            View::Result => result.is_some(),
            View::Input => false,
        };
        if ready {
            changed |= self.strip(ui, canvas_rect, view, session);
        }
        changed
    }

    fn shape_menu(&mut self, ui: &mut Ui, canvas: &Canvas, session: &mut Session) -> bool {
        let Some(m) = &mut self.menu else { return false };
        let t = theme::tokens(ui.ctx());
        let removed = session.background().is_hidden(m.shape);
        let c = m.result.v.colors[m.shape];
        let (x, y) = (m.point[0] as i32, m.point[1] as i32);
        // The other colours of the segmentation (a shape recoloured to one joins its areas).
        let own = session.color_at(x, y);
        let mut swatches: Vec<[u8; 4]> = Vec::new();
        if let Some(seg) = session.segmentation() {
            for k in 0..seg.seg.segments.len() {
                let k = seg.seg.segment_color(k);
                if swatches.len() < MAX_SWATCHES && Some(k) != own && !swatches.contains(&k) {
                    swatches.push(k);
                }
            }
        }
        let mut action = None;
        let mut close = false;
        egui::Area::new(egui::Id::new("shape_menu"))
            .fixed_pos(canvas.to_screen(m.point) + vec2(14.0, 14.0))
            .order(egui::Order::Foreground)
            .constrain(true)
            .show(ui.ctx(), |ui| {
                egui::Frame::new()
                    .fill(t.panel)
                    .stroke(Stroke::new(1.0, t.border))
                    .corner_radius(CornerRadius::same(8))
                    .inner_margin(Margin::same(10))
                    .show(ui, |ui| {
                        ui.set_width(220.0);
                        ui.horizontal(|ui| {
                            swatch(ui, c, 16.0);
                            ui.label(RichText::new("Shape").strong());
                            ui.label(RichText::new(&hex_rgba(c)[..7]).monospace().color(t.muted));
                            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                                if ui.small_button("×").on_hover_text("Close (Esc)").clicked() {
                                    close = true;
                                }
                            });
                        });
                        ui.add_space(6.0);
                        let (label, tip) = if removed {
                            ("Restore shape", "Put the shape back into the export")
                        } else {
                            ("Remove shape", "Leave the shape out of the export (e.g. the background)")
                        };
                        if ui
                            .add_sized([ui.available_width(), 24.0], egui::Button::new(label))
                            .on_hover_text(tip)
                            .clicked()
                        {
                            action = Some(Edit::Remove(!removed));
                        }
                        if ui
                            .add_sized([ui.available_width(), 24.0], egui::Button::new("Merge into neighbours"))
                            .on_hover_text(Tool::Merge.tip())
                            .clicked()
                        {
                            action = Some(Edit::Segments(Action::Zap { x, y }));
                        }
                        ui.add_space(6.0);
                        ui.label(RichText::new("Recolour").color(t.muted));
                        ui.horizontal_wrapped(|ui| {
                            ui.spacing_mut().item_spacing = vec2(4.0, 4.0);
                            for &k in &swatches {
                                if swatch(ui, k, 18.0).on_hover_text(&hex_rgba(k)[..7]).clicked() {
                                    action = Some(Edit::Segments(Action::Fill { x, y, color: k }));
                                }
                            }
                        });
                        ui.horizontal(|ui| {
                            egui::color_picker::color_edit_button_srgba(
                                ui,
                                &mut m.custom,
                                egui::color_picker::Alpha::OnlyBlend,
                            );
                            if accent_button(ui, "Apply").on_hover_text("Recolour the shape with this colour").clicked()
                            {
                                let color = m.custom.to_srgba_unmultiplied();
                                action = Some(Edit::Segments(Action::Fill { x, y, color }));
                            }
                        });
                    });
            });
        let point = m.point;
        if close || action.is_some() {
            self.menu = None;
        }
        match action {
            Some(Edit::Remove(remove)) => session.set_shape_removed(point, remove),
            Some(Edit::Segments(a)) => session.edit_segments(a),
            None => false,
        }
    }

    /// The strip at the bottom left of the canvas.
    fn strip(&mut self, ui: &mut Ui, canvas: Rect, view: View, session: &mut Session) -> bool {
        let t = theme::tokens(ui.ctx());
        let mut changed = false;
        egui::Area::new(egui::Id::new("edit_strip"))
            .pivot(egui::Align2::LEFT_BOTTOM)
            .fixed_pos(canvas.left_bottom() + vec2(12.0, -12.0))
            .order(egui::Order::Foreground)
            .show(ui.ctx(), |ui| {
                egui::Frame::new()
                    .fill(t.panel)
                    .stroke(Stroke::new(1.0, t.border))
                    .corner_radius(CornerRadius::same(8))
                    .inner_margin(Margin::same(6))
                    .show(ui, |ui| {
                        ui.horizontal(|ui| {
                            changed = history_buttons(ui, session);
                            ui.separator();
                            match view {
                                View::Segments => self.paint_strip(ui, session),
                                _ => {
                                    let removed = session.background().hidden().iter().filter(|&&h| h).count();
                                    let note = match removed {
                                        0 => "Click a shape to remove, recolour or merge it".to_owned(),
                                        n => format!("{n} removed · not exported · click a shape for options"),
                                    };
                                    ui.label(RichText::new(note).color(t.muted));
                                }
                            }
                        });
                    });
            });
        changed
    }

    fn paint_strip(&mut self, ui: &mut Ui, session: &Session) {
        let t = theme::tokens(ui.ctx());
        ui.toggle_value(&mut self.paint, "Paint pixels")
            .on_hover_text("Fix the segmentation pixel by pixel (E); the result is traced again");
        if !self.paint {
            ui.label(RichText::new("The colour areas the result is traced from").color(t.muted));
            return;
        }
        for tool in TOOLS {
            let tip = match tool.key() {
                Some(k) => format!("{} ({})", tool.tip(), k.name()),
                None => tool.tip().to_owned(),
            };
            if ui.add(egui::Button::selectable(self.tool == tool, tool.label())).on_hover_text(tip).clicked() {
                self.select(tool);
            }
        }
        ui.separator();
        // The swatch shows the colour under the picker while picking.
        let shown = self.hover_color.or(self.color);
        let (rect, resp) = ui.allocate_exact_size(vec2(36.0, 20.0), egui::Sense::hover());
        let fill = shown.map_or(Color32::TRANSPARENT, |c| Color32::from_rgba_unmultiplied(c[0], c[1], c[2], c[3]));
        ui.painter().rect_filled(rect, CornerRadius::same(3), fill);
        ui.painter().rect_stroke(rect, CornerRadius::same(3), Stroke::new(1.0, t.border), egui::StrokeKind::Inside);
        if let Some(c) = shown {
            resp.on_hover_text(format!("Current colour {}", &hex_rgba(c)[..7]));
        }
        if session.seg_editor().is_some_and(|e| e.has_edits()) {
            ui.label(RichText::new("Re-segmenting discards these edits").color(t.muted));
        }
    }

    /// Default the swatch to the segmentation's first colour.
    pub fn sync_color(&mut self, session: &Session) {
        if self.color.is_none()
            && let Some(s) = session.segmentation()
            && !s.seg.segments.is_empty()
        {
            self.color = Some(s.seg.segment_color(0));
        }
    }
}

/// A shape menu choice.
enum Edit {
    /// Remove (true) or restore the shape.
    Remove(bool),
    Segments(Action),
}

/// Undo, redo and revert over both kinds of edit.
fn history_buttons(ui: &mut Ui, session: &mut Session) -> bool {
    let mut changed = false;
    if ui.add_enabled(session.can_undo(), egui::Button::new("Undo")).on_hover_text("Undo (Ctrl+Z)").clicked() {
        changed |= session.undo();
    }
    if ui.add_enabled(session.can_redo(), egui::Button::new("Redo")).on_hover_text("Redo (Ctrl+Shift+Z)").clicked() {
        changed |= session.redo();
    }
    if ui
        .add_enabled(session.can_undo(), egui::Button::new("Revert"))
        .on_hover_text("Undo every edit (Redo brings them back)")
        .clicked()
    {
        changed |= session.revert();
    }
    changed
}

/// A clickable colour square.
fn swatch(ui: &mut Ui, c: [u8; 4], size: f32) -> egui::Response {
    let t = theme::tokens(ui.ctx());
    let (rect, resp) = ui.allocate_exact_size(vec2(size, size), egui::Sense::click());
    let stroke = if resp.hovered() { Stroke::new(2.0, t.accent) } else { Stroke::new(1.0, t.border) };
    ui.painter().rect_filled(rect, CornerRadius::same(3), Color32::from_rgba_unmultiplied(c[0], c[1], c[2], c[3]));
    ui.painter().rect_stroke(rect, CornerRadius::same(3), stroke, egui::StrokeKind::Inside);
    resp
}

/// A shape outline over a white halo.
fn draw_outline(ui: &Ui, clip: Rect, canvas: &Canvas, outlines: &[Vec<[f64; 2]>], color: Color32) {
    let painter = ui.painter_at(clip);
    for line in outlines {
        let pts = closed(canvas, line);
        painter.add(Shape::line(pts.clone(), Stroke::new(3.0, Color32::WHITE)));
        painter.add(Shape::line(pts, Stroke::new(1.5, color)));
    }
}

fn closed(canvas: &Canvas, line: &[[f64; 2]]) -> Vec<Pos2> {
    let mut pts: Vec<Pos2> = line.iter().map(|&p| canvas.to_screen(p)).collect();
    if let Some(&first) = pts.first() {
        pts.push(first);
    }
    pts
}

/// Removed shapes in the Result view: a white line under red 5-on/5-off dashes, no fill
/// (cosmetic 1 px pens).
pub fn draw_removed(ui: &Ui, clip: Rect, canvas: &Canvas, outlines: &[Vec<[f64; 2]>]) {
    let painter = ui.painter_at(clip);
    for line in outlines {
        let pts = closed(canvas, line);
        painter.add(Shape::line(pts.clone(), Stroke::new(1.0, Color32::WHITE)));
        painter.extend(Shape::dashed_line(&pts, Stroke::new(1.0, Color32::RED), 5.0, 5.0));
    }
}
