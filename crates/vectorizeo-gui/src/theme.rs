//! Visual style: light and dark colour tokens, spacing and type, and a few small widgets built
//! on them (segmented control, section header, badge).

use eframe::egui::{
    self, Button, Color32, CornerRadius, FontFamily, FontId, Frame, Margin, Response, RichText, Stroke, TextStyle,
    Theme, Ui, Vec2,
};

/// Colour tokens of one theme.
#[derive(Clone, Copy, Debug)]
pub struct Tokens {
    /// Behind the image.
    pub canvas: Color32,
    /// Panels and bars.
    pub panel: Color32,
    /// Controls, cards and the segmented-control track.
    pub surface: Color32,
    pub surface_hover: Color32,
    pub border: Color32,
    pub text: Color32,
    pub muted: Color32,
    pub accent: Color32,
    /// Text on accent fills.
    pub on_accent: Color32,
    /// Accent tint for selected rows.
    pub accent_soft: Color32,
    pub danger: Color32,
    pub checker: [Color32; 2],
}

pub const LIGHT: Tokens = Tokens {
    canvas: Color32::from_rgb(0xec, 0xed, 0xf1),
    panel: Color32::from_rgb(0xfb, 0xfb, 0xfc),
    surface: Color32::from_rgb(0xf0, 0xf1, 0xf4),
    surface_hover: Color32::from_rgb(0xe6, 0xe8, 0xed),
    border: Color32::from_rgb(0xe0, 0xe2, 0xe8),
    text: Color32::from_rgb(0x1b, 0x1d, 0x22),
    muted: Color32::from_rgb(0x6b, 0x70, 0x80),
    accent: Color32::from_rgb(0x4f, 0x5b, 0xd5),
    on_accent: Color32::WHITE,
    accent_soft: Color32::from_rgb(0xe7, 0xe9, 0xfb),
    danger: Color32::from_rgb(0xd1, 0x43, 0x43),
    checker: [Color32::from_rgb(0xff, 0xff, 0xff), Color32::from_rgb(0xe4, 0xe5, 0xea)],
};

pub const DARK: Tokens = Tokens {
    canvas: Color32::from_rgb(0x12, 0x13, 0x16),
    panel: Color32::from_rgb(0x1b, 0x1c, 0x20),
    surface: Color32::from_rgb(0x25, 0x26, 0x2b),
    surface_hover: Color32::from_rgb(0x2f, 0x30, 0x36),
    border: Color32::from_rgb(0x31, 0x33, 0x39),
    text: Color32::from_rgb(0xe8, 0xe9, 0xec),
    muted: Color32::from_rgb(0x9a, 0x9d, 0xa8),
    accent: Color32::from_rgb(0x7c, 0x86, 0xff),
    on_accent: Color32::from_rgb(0x10, 0x11, 0x20),
    accent_soft: Color32::from_rgb(0x2a, 0x2d, 0x4a),
    danger: Color32::from_rgb(0xf0, 0x6a, 0x6a),
    checker: [Color32::from_rgb(0x2a, 0x2b, 0x30), Color32::from_rgb(0x22, 0x23, 0x27)],
};

pub const RADIUS: u8 = 6;
pub const CARD_RADIUS: u8 = 10;

/// The tokens of the theme egui is currently showing.
pub fn tokens(ctx: &egui::Context) -> Tokens {
    match ctx.theme() {
        Theme::Light => LIGHT,
        Theme::Dark => DARK,
    }
}

/// Install both styles; egui picks one from the system or the user's choice.
pub fn install(ctx: &egui::Context) {
    ctx.set_style_of(Theme::Light, style(Theme::Light, &LIGHT));
    ctx.set_style_of(Theme::Dark, style(Theme::Dark, &DARK));
}

fn style(theme: Theme, t: &Tokens) -> egui::Style {
    let mut style = egui::Style { visuals: theme.default_visuals(), ..Default::default() };
    style.text_styles = [
        (TextStyle::Small, FontId::new(11.5, FontFamily::Proportional)),
        (TextStyle::Body, FontId::new(13.5, FontFamily::Proportional)),
        (TextStyle::Button, FontId::new(13.5, FontFamily::Proportional)),
        (TextStyle::Heading, FontId::new(17.0, FontFamily::Proportional)),
        (TextStyle::Monospace, FontId::new(12.5, FontFamily::Monospace)),
    ]
    .into();
    // Rows are clickable cards; selectable text would fight with them.
    style.interaction.selectable_labels = false;
    let s = &mut style.spacing;
    s.item_spacing = Vec2::new(8.0, 8.0);
    s.button_padding = Vec2::new(10.0, 5.0);
    s.interact_size = Vec2::new(28.0, 28.0);
    s.slider_width = 150.0;
    s.window_margin = Margin::same(12);
    s.menu_margin = Margin::same(6);

    let v = &mut style.visuals;
    v.panel_fill = t.panel;
    v.window_fill = t.panel;
    v.window_stroke = Stroke::new(1.0, t.border);
    v.window_corner_radius = CornerRadius::same(CARD_RADIUS);
    v.menu_corner_radius = CornerRadius::same(8);
    v.extreme_bg_color = t.surface;
    v.faint_bg_color = t.surface;
    v.override_text_color = None;
    v.hyperlink_color = t.accent;
    v.selection.bg_fill = t.accent_soft;
    v.selection.stroke = Stroke::new(1.0, t.accent);
    v.slider_trailing_fill = true;
    v.collapsing_header_frame = false;
    v.indent_has_left_vline = false;
    let r = CornerRadius::same(RADIUS);
    let w = &mut v.widgets;
    w.noninteractive.bg_fill = t.panel;
    w.noninteractive.weak_bg_fill = t.panel;
    w.noninteractive.bg_stroke = Stroke::new(1.0, t.border);
    w.noninteractive.fg_stroke = Stroke::new(1.0, t.text);
    w.noninteractive.corner_radius = r;
    for (wv, fill, stroke) in [
        (&mut w.inactive, t.surface, Stroke::new(1.0, t.border)),
        (&mut w.hovered, t.surface_hover, Stroke::new(1.0, t.border)),
        (&mut w.active, t.surface_hover, Stroke::new(1.0, t.accent)),
        (&mut w.open, t.surface_hover, Stroke::new(1.0, t.border)),
    ] {
        wv.bg_fill = fill;
        wv.weak_bg_fill = fill;
        wv.bg_stroke = stroke;
        wv.fg_stroke = Stroke::new(1.5, t.text);
        wv.corner_radius = r;
        wv.expansion = 0.0;
    }
    // Slider rails.
    w.inactive.bg_fill = t.border;
    style
}

/// A row of mutually exclusive options on a rounded track.
pub fn segmented<T: PartialEq + Copy>(ui: &mut Ui, value: &mut T, options: &[(T, &str)]) -> Response {
    let t = tokens(ui.ctx());
    let inner = Frame::new()
        .fill(t.surface)
        .stroke(Stroke::new(1.0, t.border))
        .corner_radius(CornerRadius::same(8))
        .inner_margin(Margin::same(2))
        .show(ui, |ui| {
            ui.spacing_mut().item_spacing.x = 2.0;
            ui.horizontal(|ui| {
                let mut changed: Option<Response> = None;
                // Keep the reading order inside right-to-left layouts.
                let rtl = ui.layout().prefer_right_to_left();
                let ordered: Vec<_> = if rtl { options.iter().rev().collect() } else { options.iter().collect() };
                for &(v, label) in ordered {
                    let sel = *value == v;
                    let text = RichText::new(label).color(if sel { t.text } else { t.muted });
                    let mut b = Button::new(text).corner_radius(CornerRadius::same(RADIUS));
                    b = if sel {
                        b.fill(t.panel).stroke(Stroke::new(1.0, t.border))
                    } else {
                        b.frame_when_inactive(false)
                    };
                    let r = ui.add(b);
                    if r.clicked() && !sel {
                        *value = v;
                        let mut r = r;
                        r.mark_changed();
                        changed = Some(r);
                    }
                }
                changed
            })
            .inner
        });
    inner.inner.unwrap_or(inner.response)
}

/// The prominent action button.
pub fn accent_button(ui: &mut Ui, label: &str) -> Response {
    let t = tokens(ui.ctx());
    ui.add(
        Button::new(RichText::new(label).color(t.on_accent).strong())
            .fill(t.accent)
            .stroke(Stroke::NONE)
            .corner_radius(CornerRadius::same(RADIUS)),
    )
}

/// An accent button with a ▾ part that opens `menu`: returns the main part's response and the
/// menu's result while it is open.
pub fn accent_split_button<R>(ui: &mut Ui, label: &str, menu: impl FnOnce(&mut Ui) -> R) -> (Response, Option<R>) {
    let t = tokens(ui.ctx());
    let r = RADIUS;
    ui.horizontal(|ui| {
        ui.spacing_mut().item_spacing.x = 1.0;
        let main = ui.add(
            Button::new(RichText::new(label).color(t.on_accent).strong())
                .fill(t.accent)
                .stroke(Stroke::NONE)
                .corner_radius(CornerRadius { nw: r, sw: r, ne: 0, se: 0 }),
        );
        let arrow = Button::new(RichText::new("⏷").color(t.on_accent))
            .fill(t.accent)
            .stroke(Stroke::NONE)
            .corner_radius(CornerRadius { nw: 0, sw: 0, ne: r, se: r });
        let (_, open) = egui::containers::menu::MenuButton::from_button(arrow).ui(ui, menu);
        (main, open.map(|o| o.inner))
    })
    .inner
}

/// Uppercase caption above a group of controls.
pub fn caption(ui: &mut Ui, text: &str) {
    let t = tokens(ui.ctx());
    ui.label(RichText::new(text.to_uppercase()).size(10.5).color(t.muted).strong());
}

/// The dissolving left arm of the logo, one convex polygon per (possibly cut) pixel, in the 256-unit
/// space of `assets/logo-small.svg`: the cut with fewer, larger pixels that survives at toolbar size.
const LOGO_PIXELS: [&[(f32, f32)]; 20] = [
    &[(35.8, 45.1), (42.6, 45.1), (42.6, 54.1), (40.5, 54.1)],
    &[(52.0, 44.9), (61.6, 44.9), (61.6, 54.4), (52.0, 54.4)],
    &[(70.2, 44.4), (80.7, 44.4), (80.7, 54.9), (70.2, 54.9)],
    &[(88.4, 43.9), (91.3, 43.9), (97.7, 55.4), (88.4, 55.4)],
    &[(45.4, 63.7), (52.3, 63.7), (52.3, 72.9), (50.1, 72.9)],
    &[(61.4, 63.2), (71.5, 63.2), (71.5, 73.3), (61.4, 73.3)],
    &[(79.6, 62.8), (90.6, 62.8), (90.6, 73.8), (79.6, 73.8)],
    &[(97.8, 62.4), (101.6, 62.4), (108.2, 74.2), (97.8, 74.2)],
    &[(54.8, 82.0), (62.3, 82.0), (62.3, 91.9), (59.9, 91.9)],
    &[(70.7, 81.6), (81.3, 81.6), (81.3, 92.3), (70.7, 92.3)],
    &[(89.0, 81.2), (100.4, 81.2), (100.4, 92.7), (89.0, 92.7)],
    &[(64.3, 100.3), (72.2, 100.3), (72.2, 110.9), (69.7, 110.9)],
    &[(80.0, 100.0), (91.2, 100.0), (91.2, 111.2), (80.0, 111.2)],
    &[(98.4, 99.7), (110.2, 99.7), (110.2, 111.5), (98.4, 111.5)],
    &[(73.7, 118.7), (82.2, 118.7), (82.2, 129.9), (79.5, 129.9)],
    &[(89.4, 118.4), (101.1, 118.4), (101.1, 130.2), (89.4, 130.2)],
    &[(83.1, 137.0), (92.1, 137.0), (92.1, 148.9), (89.2, 148.9)],
    &[(98.7, 136.8), (111.0, 136.8), (111.0, 149.1), (98.7, 149.1)],
    &[(92.6, 155.3), (102.1, 155.3), (102.1, 167.9), (99.0, 167.9)],
    &[(102.0, 173.7), (112.0, 173.7), (112.0, 186.9), (108.8, 186.9)],
];

/// The app mark: `assets/logo-small.svg` (a clean vector arm and a left arm dissolving into pixels)
/// on an accent tile, painted as shapes so it stays crisp at any scale.
pub fn logo(ui: &mut Ui) {
    let t = tokens(ui.ctx());
    let (rect, _) = ui.allocate_exact_size(Vec2::splat(26.0), egui::Sense::hover());
    let p = ui.painter();
    p.rect_filled(rect, CornerRadius::same(7), t.accent);
    let scale = rect.width() * 0.74 / 183.0;
    let at = |&(x, y): &(f32, f32)| rect.center() + egui::vec2((x - 124.7) * scale, (y - 129.0) * scale);
    let poly = |pts: &[(f32, f32)]| {
        p.add(egui::Shape::convex_polygon(pts.iter().map(at).collect(), t.on_accent, Stroke::NONE));
    };
    // The smooth arm, and the strip beside it that the pixels end against.
    poly(&[(124.69, 104.04), (160.0, 40.0), (216.0, 40.0), (124.73, 217.88), (117.51, 203.88)]);
    poly(&[(117.53, 91.04), (124.69, 104.04), (124.69, 203.88), (117.51, 203.88)]);
    for pixel in LOGO_PIXELS {
        poly(pixel);
    }
}

/// A small pill, e.g. "Recommended".
pub fn badge(ui: &mut Ui, text: &str) -> Response {
    let t = tokens(ui.ctx());
    Frame::new()
        .fill(t.accent_soft)
        .corner_radius(CornerRadius::same(8))
        .inner_margin(Margin::symmetric(6, 1))
        .show(ui, |ui| ui.label(RichText::new(text).size(10.5).color(t.accent).strong()))
        .response
}

/// A selectable card row: title, optional subtitle, optional "Recommended" badge, and anything
/// `extra` adds at the right. The whole row is clickable.
pub fn option_row(
    ui: &mut Ui,
    selected: bool,
    title: &str,
    subtitle: Option<&str>,
    recommended: bool,
    extra: impl FnOnce(&mut Ui),
) -> Response {
    let t = tokens(ui.ctx());
    let (fill, stroke) = if selected { (t.accent_soft, t.accent) } else { (t.panel, t.border) };
    let inner = Frame::new()
        .fill(fill)
        .stroke(Stroke::new(1.0, stroke))
        .corner_radius(CornerRadius::same(8))
        .inner_margin(Margin::symmetric(10, 7))
        .show(ui, |ui| {
            ui.set_width(ui.available_width());
            ui.horizontal(|ui| {
                radio_dot(ui, selected);
                ui.vertical(|ui| {
                    ui.spacing_mut().item_spacing.y = 1.0;
                    ui.horizontal(|ui| {
                        ui.label(RichText::new(title).color(t.text));
                        if recommended {
                            badge(ui, "Recommended");
                        }
                    });
                    if let Some(sub) = subtitle {
                        ui.label(RichText::new(sub).size(11.5).color(t.muted));
                    }
                });
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), extra);
            });
        });
    let response = ui.interact(inner.response.rect, inner.response.id.with("row"), egui::Sense::click());
    if response.hovered() && !selected {
        ui.painter().rect_stroke(
            inner.response.rect,
            CornerRadius::same(8),
            Stroke::new(1.0, t.muted),
            egui::StrokeKind::Inside,
        );
    }
    response
}

fn radio_dot(ui: &mut Ui, on: bool) {
    let t = tokens(ui.ctx());
    let (rect, _) = ui.allocate_exact_size(Vec2::splat(14.0), egui::Sense::hover());
    let c = rect.center();
    if on {
        ui.painter().circle_filled(c, 7.0, t.accent);
        ui.painter().circle_filled(c, 2.5, t.on_accent);
    } else {
        ui.painter().circle_stroke(c, 6.5, Stroke::new(1.0, t.muted));
    }
}

/// A strip of colour swatches (RGBA, straight alpha) over a checkerboard.
pub fn swatches(ui: &mut Ui, colors: &[[u8; 4]], size: f32) -> Response {
    let t = tokens(ui.ctx());
    let n = colors.len().max(1) as f32;
    // Shrink to the space left (a 12-colour strip in a narrow panel).
    let size = size.min((ui.available_width() + 2.0) / n - 2.0).max(4.0);
    let (rect, response) = ui.allocate_exact_size(egui::vec2(n * (size + 2.0) - 2.0, size), egui::Sense::hover());
    let p = ui.painter();
    for (i, c) in colors.iter().enumerate() {
        let r = egui::Rect::from_min_size(rect.min + egui::vec2(i as f32 * (size + 2.0), 0.0), Vec2::splat(size));
        if c[3] < 255 {
            p.rect_filled(r, CornerRadius::same(3), t.checker[1]);
        }
        p.rect_filled(r, CornerRadius::same(3), Color32::from_rgba_unmultiplied(c[0], c[1], c[2], c[3]));
        p.rect_stroke(r, CornerRadius::same(3), Stroke::new(1.0, t.border), egui::StrokeKind::Inside);
    }
    response
}

/// A collapsible group of settings with an uppercase caption.
pub fn section<R>(ui: &mut Ui, title: &str, add: impl FnOnce(&mut Ui) -> R) -> Option<R> {
    let t = tokens(ui.ctx());
    ui.add_space(6.0);
    let r = egui::CollapsingHeader::new(RichText::new(title.to_uppercase()).size(10.5).color(t.muted).strong())
        .id_salt(title)
        .default_open(true)
        .show(ui, |ui| {
            ui.add_space(2.0);
            add(ui)
        });
    r.body_returned
}
