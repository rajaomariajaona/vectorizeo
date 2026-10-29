//! Edit mode: Edit Segmentation and Remove Background as a floating tool
//! strip over the canvas — segmentation tools in the Segments view, Remove Background in the
//! Result view. Keys (no modifier): Z undo, Y redo, T revert, A pan, D zap /
//! remove, F fill, X pencil, C colour; Shift pans for a moment.

use eframe::egui::{self, Color32, CornerRadius, Key, Margin, Pos2, Rect, RichText, Shape, Stroke, Ui, vec2};
use vectorizeo_pipeline::edit::Action;
use vectorizeo_pipeline::session::Session;

use crate::app::View;
use crate::canvas::{Canvas, Pointer};
use crate::theme;

/// Tool buttons (`VmMainView` input modes: pan 0, pencil 3, colour 4, fill 5, zap 6, remove 7).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Tool {
    Pan,
    Zap,
    Fill,
    Pencil,
    Picker,
    Remove,
}

impl Tool {
    fn label(self) -> &'static str {
        match self {
            Tool::Pan => "Pan",
            Tool::Zap => "Zap",
            Tool::Fill => "Fill",
            Tool::Pencil => "Pencil",
            Tool::Picker => "Color",
            Tool::Remove => "Remove",
        }
    }

    fn key(self) -> Key {
        match self {
            Tool::Pan => Key::A,
            Tool::Zap | Tool::Remove => Key::D,
            Tool::Fill => Key::F,
            Tool::Pencil => Key::X,
            Tool::Picker => Key::C,
        }
    }

    fn tip(self) -> &'static str {
        match self {
            Tool::Pan => "Drag to move the image",
            Tool::Zap => "Split a segment and give each pixel the closest neighbouring colour (anti-aliasing remnants)",
            Tool::Fill => "Fill the connected area of one colour with the current colour",
            Tool::Pencil => "Draw with the current colour",
            Tool::Picker => "Pick the current colour from the segments, then return to the previous tool",
            Tool::Remove => "Click shapes to remove them from the result; click a removed shape to restore it",
        }
    }
}

const SEG_TOOLS: [Tool; 5] = [Tool::Pan, Tool::Zap, Tool::Fill, Tool::Pencil, Tool::Picker];
const BG_TOOLS: [Tool; 2] = [Tool::Pan, Tool::Remove];

/// What the history buttons act on.
#[derive(Clone, Copy, PartialEq, Eq)]
enum History {
    Undo,
    Redo,
    Revert,
}

pub struct EditTools {
    pub on: bool,
    seg_tool: Tool,
    /// The tool the colour picker returns to.
    prev_tool: Tool,
    bg_tool: Tool,
    /// The pencil and fill colour (the swatch, palette[0] after a segmentation).
    pub color: Option<[u8; 4]>,
    /// The colour under the picker.
    hover_color: Option<[u8; 4]>,
    /// Previous pencil position while the button is held.
    last: Option<[f64; 2]>,
}

impl Default for EditTools {
    fn default() -> Self {
        Self {
            on: false,
            seg_tool: Tool::Pan,
            prev_tool: Tool::Pan,
            bg_tool: Tool::Remove,
            color: None,
            hover_color: None,
            last: None,
        }
    }
}

impl EditTools {
    /// The tool the canvas hands the pointer to in `view`, if any.
    pub fn active(&self, view: View) -> Option<Tool> {
        let tool = match view {
            View::Segments if self.on => self.seg_tool,
            View::Result if self.on => self.bg_tool,
            _ => return None,
        };
        (tool != Tool::Pan).then_some(tool)
    }

    /// Clicking the selected tool again goes back to Pan.
    fn select(&mut self, view: View, tool: Tool) {
        let cur = if view == View::Result { &mut self.bg_tool } else { &mut self.seg_tool };
        if *cur == tool && tool != Tool::Pan {
            *cur = Tool::Pan;
        } else {
            if tool == Tool::Picker && *cur != Tool::Picker {
                self.prev_tool = *cur;
            }
            *cur = tool;
        }
        self.last = None;
    }

    /// Edit-mode keys. Returns whether a key was used (F and C then do not fit / compare) and
    /// whether an undo/redo/revert changed something.
    pub fn keys(&mut self, ctx: &egui::Context, view: View, session: &mut Session) -> (bool, bool) {
        if !self.on || !matches!(view, View::Segments | View::Result) {
            return (false, false);
        }
        let tools: &[Tool] = if view == View::Result { &BG_TOOLS } else { &SEG_TOOLS };
        let mut used = false;
        let mut changed = false;
        ctx.input(|i| {
            if !i.modifiers.is_none() {
                return;
            }
            for &t in tools {
                if i.key_pressed(t.key()) {
                    self.select(view, t);
                    used = true;
                }
            }
            for (key, h) in [(Key::Z, History::Undo), (Key::Y, History::Redo), (Key::T, History::Revert)] {
                if i.key_pressed(key) {
                    changed |= history(view, session, h);
                    used = true;
                }
            }
        });
        (used, changed)
    }

    /// Apply the pointer to the active tool. Returns whether the segmentation or the removed
    /// shapes changed.
    pub fn pointer(&mut self, view: View, p: Pointer, session: &mut Session) -> bool {
        self.hover_color = None;
        let Some(tool) = self.active(view) else { return false };
        let Some(pos) = p.pos else { return false };
        let px = pos.map(|v| v as i32);
        match tool {
            Tool::Remove => {
                if !p.pressed {
                    return false;
                }
                let Some(r) = session.result().cloned() else { return false };
                // The shape under the point, if any.
                let Some(s) = r.v.graph.shape_at(&r.seg.seg.labels, pos[0], pos[1]) else { return false };
                session.background_mut().toggle(s);
                true
            }
            Tool::Picker => {
                let Some(e) = session.seg_editor_mut() else { return false };
                self.hover_color = e.color_at(px[0], px[1]);
                if p.released {
                    if let Some(c) = self.hover_color {
                        self.color = Some(c);
                    }
                    self.seg_tool = self.prev_tool;
                }
                false
            }
            Tool::Pencil => {
                let color = self.color;
                let Some(e) = session.seg_editor_mut() else { return false };
                let Some(color) = color else { return false };
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
                e.apply(Action::Pencil { from, to: pos, color })
            }
            Tool::Fill | Tool::Zap => {
                if !p.pressed {
                    return false;
                }
                let color = self.color;
                let Some(e) = session.seg_editor_mut() else { return false };
                let (x, y) = (px[0], px[1]);
                match (tool, color) {
                    (Tool::Fill, Some(color)) => e.apply(Action::Fill { x, y, color }),
                    (Tool::Zap, _) => e.apply(Action::Zap { x, y }),
                    _ => false,
                }
            }
            Tool::Pan => false,
        }
    }

    /// The floating tool strip at the top left of the canvas.
    pub fn toolbar(&mut self, ui: &mut Ui, canvas: Rect, view: View, session: &mut Session) -> bool {
        if !self.on {
            return false;
        }
        let t = theme::tokens(ui.ctx());
        let mut changed = false;
        egui::Area::new(egui::Id::new("edit_tools"))
            .fixed_pos(canvas.left_top() + vec2(12.0, 12.0))
            .order(egui::Order::Foreground)
            .show(ui.ctx(), |ui| {
                egui::Frame::new()
                    .fill(t.panel)
                    .stroke(Stroke::new(1.0, t.border))
                    .corner_radius(CornerRadius::same(8))
                    .inner_margin(Margin::same(6))
                    .show(ui, |ui| {
                        ui.horizontal(|ui| match view {
                            View::Segments => changed = self.seg_strip(ui, session),
                            View::Result => changed = self.bg_strip(ui, session),
                            View::Input => {
                                ui.label(RichText::new("Edit: open the Segments or Result view").color(t.muted));
                            }
                        });
                    });
            });
        changed
    }

    fn history_buttons(&mut self, ui: &mut Ui, view: View, session: &mut Session, can: (bool, bool)) -> bool {
        let mut changed = false;
        for (label, key, h, enabled) in [
            ("Undo", "Z", History::Undo, can.0),
            ("Redo", "Y", History::Redo, can.1),
            ("Revert", "T", History::Revert, can.0),
        ] {
            let b = ui.add_enabled(enabled, egui::Button::new(label)).on_hover_text(format!("{label} ({key})"));
            if b.clicked() {
                changed |= history(view, session, h);
            }
        }
        ui.separator();
        changed
    }

    fn tool_buttons(&mut self, ui: &mut Ui, view: View, tools: &[Tool], enabled: bool) {
        let cur = if view == View::Result { self.bg_tool } else { self.seg_tool };
        for &tool in tools {
            let key = format!("{:?}", tool.key());
            let b = ui
                .add_enabled(enabled, egui::Button::selectable(cur == tool, tool.label()))
                .on_hover_text(format!("{} ({key})", tool.tip()));
            if b.clicked() {
                self.select(view, tool);
            }
        }
    }

    fn seg_strip(&mut self, ui: &mut Ui, session: &mut Session) -> bool {
        let t = theme::tokens(ui.ctx());
        let ready = session.segmentation().is_some();
        let (undo, redo) = session.seg_editor().map_or((false, false), |e| (e.can_undo(), e.can_redo()));
        let changed = self.history_buttons(ui, View::Segments, session, (undo, redo));
        self.tool_buttons(ui, View::Segments, &SEG_TOOLS, ready);
        ui.separator();
        // The swatch shows the colour under the picker while picking.
        let shown = self.hover_color.or(self.color);
        let (rect, resp) = ui.allocate_exact_size(vec2(36.0, 20.0), egui::Sense::hover());
        let fill = shown.map_or(Color32::TRANSPARENT, |c| Color32::from_rgba_unmultiplied(c[0], c[1], c[2], c[3]));
        ui.painter().rect_filled(rect, CornerRadius::same(3), fill);
        ui.painter().rect_stroke(rect, CornerRadius::same(3), Stroke::new(1.0, t.border), egui::StrokeKind::Inside);
        if let Some(c) = shown {
            resp.on_hover_text(format!("Current colour #{:02x}{:02x}{:02x}", c[0], c[1], c[2]));
        }
        if !ready {
            ui.label(RichText::new("Waiting for the segmentation…").color(t.muted));
        } else if session.seg_editor().is_some_and(|e| e.has_edits()) {
            ui.label(RichText::new("Re-segmenting discards these edits").color(t.muted));
        }
        changed
    }

    fn bg_strip(&mut self, ui: &mut Ui, session: &mut Session) -> bool {
        let t = theme::tokens(ui.ctx());
        let ready = session.result().is_some();
        let b = session.background();
        let changed = self.history_buttons(ui, View::Result, session, (b.can_undo(), b.can_redo()));
        self.tool_buttons(ui, View::Result, &BG_TOOLS, ready);
        let removed = session.background().hidden().iter().filter(|&&h| h).count();
        let note = if !ready {
            "Waiting for the result…".to_owned()
        } else if removed > 0 {
            format!("{removed} removed · not exported")
        } else {
            "Click shapes to remove them".to_owned()
        };
        ui.label(RichText::new(note).color(t.muted));
        changed
    }

    /// Default the swatch to the segmentation'sfirst colour.
    pub fn sync_color(&mut self, session: &Session) {
        if self.color.is_none()
            && let Some(s) = session.segmentation()
            && !s.seg.segments.is_empty()
        {
            self.color = Some(s.seg.segment_color(0));
        }
    }
}

fn history(view: View, session: &mut Session, h: History) -> bool {
    match view {
        View::Segments => {
            let Some(e) = session.seg_editor_mut() else { return false };
            let before = e.revision();
            match h {
                History::Undo => e.undo(),
                History::Redo => e.redo(),
                History::Revert => e.revert(),
            }
            e.revision() != before
        }
        View::Result => {
            let b = session.background_mut();
            let before = b.revision();
            match h {
                History::Undo => b.undo(),
                History::Redo => b.redo(),
                History::Revert => b.revert(),
            }
            b.revision() != before
        }
        View::Input => false,
    }
}

/// Removed shapes in the Result view: a white line under red 5-on/5-off dashes, no fill
/// (cosmetic 1 px pens).
pub fn draw_removed(ui: &Ui, clip: Rect, canvas: &Canvas, outlines: &[Vec<[f64; 2]>]) {
    let painter = ui.painter_at(clip);
    for line in outlines {
        let mut pts: Vec<Pos2> = line.iter().map(|&p| canvas.to_screen(p)).collect();
        if let Some(&first) = pts.first() {
            pts.push(first);
        }
        painter.add(Shape::line(pts.clone(), Stroke::new(1.0, Color32::WHITE)));
        painter.extend(Shape::dashed_line(&pts, Stroke::new(1.0, Color32::RED), 5.0, 5.0));
    }
}
