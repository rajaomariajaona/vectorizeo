//! The settings panel: every basic and advanced setting
//! (transparency, image type, quality, palette, segmentation, smoothing, export), in one list.

use eframe::egui::{self, Color32, RichText, Slider, Ui};
use vectorizeo_core::export::ShapeMode;
use vectorizeo_core::params::{AaRejection, AdvancedSettings, ImageType, Quality};
use vectorizeo_core::stages::classify::Classification;
use vectorizeo_pipeline::Flatten;
use vectorizeo_pipeline::session::{FinderSettings, PaletteChoice, Palettes, Settings, quality_name, type_name};

use crate::help;
use crate::theme::{self, option_row, section, segmented, swatches};

/// What the inspector needs besides the settings.
pub struct Inputs<'a> {
    pub class: &'a Classification,
    pub has_alpha: bool,
    /// The palette search for the current settings, once found.
    pub palettes: Option<&'a Palettes>,
}

/// An option under the pointer, previewed on the canvas.
#[derive(Clone, Debug, PartialEq)]
pub enum Hover {
    /// The input flattened this way (previewed on hover).
    Flatten(Flatten),
    /// The input with every pixel rounded to the nearest palette colour (the palette
    /// "Compare" view).
    Palette(Vec<[u8; 4]>),
}

pub fn show(ui: &mut Ui, s: &mut Settings, inp: &Inputs) -> Option<Hover> {
    let mut hover = None;
    ui.add_space(4.0);
    segmented(ui, &mut s.advanced, &[(false, "Basic"), (true, "Advanced")]).on_hover_text(
        "Basic: pick the image type, quality and palette.\nAdvanced: set the segmentation and smoothing controls.",
    );
    if inp.has_alpha {
        section(ui, "Transparency", |ui| transparency(ui, s, &mut hover));
    }
    if s.advanced {
        section(ui, "Palette finder", |ui| finder(ui, &mut s.finder));
    } else {
        section(ui, "Image type", |ui| image_type(ui, s, inp.class));
        let title = if s.image_type == ImageType::Photo { "Detail level" } else { "Input quality" };
        section(ui, title, |ui| quality(ui, s, inp.class));
    }
    section(ui, "Colours", |ui| colours(ui, s, inp.palettes, &mut hover));
    if s.advanced {
        section(ui, "Segmentation", |ui| segmentation(ui, &mut s.adv));
        section(ui, "Smoothing", |ui| smoothing(ui, &mut s.adv));
        ui.add_space(6.0);
        if ui.button("Reset advanced settings").on_hover_text("The default advanced settings").clicked() {
            s.adv = AdvancedSettings::default();
            s.finder = FinderSettings::default();
        }
    }
    section(ui, "Export", |ui| export(ui, s));
    ui.add_space(12.0);
    hover
}

fn transparency(ui: &mut Ui, s: &mut Settings, hover: &mut Option<Hover>) {
    // The custom colour is kept while another mode is selected.
    let id = ui.id().with("flatten_custom");
    let mut custom = ui.data_mut(|d| *d.get_temp_mut_or(id, [255u8; 3]));
    if let Flatten(Some(c)) = s.flatten
        && s.flatten != Flatten::WHITE
    {
        custom = c.map(|v| (v * 255.0).round() as u8)[..3].try_into().unwrap();
    }
    // QColor::setRedF etc. from the 8-bit picker value.
    let custom_flatten = |c: [u8; 3]| Flatten(Some([c[0], c[1], c[2], 255].map(|v| f32::from(v) / 255.0)));
    let is_custom = matches!(s.flatten, Flatten(Some(_))) && s.flatten != Flatten::WHITE;

    let r = option_row(ui, s.flatten == Flatten::WHITE, "Flatten onto white", None, false, |_| {});
    let r = r.on_hover_text(help::FLATTEN_WHITE);
    if r.clicked() {
        s.flatten = Flatten::WHITE;
    }
    if r.hovered() {
        *hover = Some(Hover::Flatten(Flatten::WHITE));
    }
    let mut picked = false;
    let r = option_row(ui, is_custom, "Flatten onto a colour", None, false, |ui| {
        picked = egui::color_picker::color_edit_button_srgb(ui, &mut custom).changed();
    });
    let r = r.on_hover_text(help::FLATTEN_COLOR);
    if r.clicked() || picked {
        s.flatten = custom_flatten(custom);
    }
    if r.hovered() {
        *hover = Some(Hover::Flatten(custom_flatten(custom)));
    }
    let r = option_row(ui, s.flatten == Flatten(None), "Leave transparent", None, false, |_| {});
    let r = r.on_hover_text(help::LEAVE_TRANSPARENT);
    if r.clicked() {
        s.flatten = Flatten(None);
    }
    if r.hovered() {
        *hover = Some(Hover::Flatten(Flatten(None)));
    }
    ui.data_mut(|d| d.insert_temp(id, custom));
}

fn image_type(ui: &mut Ui, s: &mut Settings, class: &Classification) {
    for (t, sub, tip) in [
        (ImageType::Photo, "Natural scenes, fastest", help::PHOTO),
        (ImageType::ArtworkBlended, "Anti-aliased logos and art", help::BLENDED),
        (ImageType::ArtworkUnblended, "Jagged, pixel-aligned edges", help::UNBLENDED),
    ] {
        let r = option_row(ui, s.image_type == t, type_name(t), Some(sub), class.image_type == t, |_| {});
        if r.on_hover_text(tip).clicked() {
            s.image_type = t;
        }
    }
}

fn quality(ui: &mut Ui, s: &mut Settings, class: &Classification) {
    let tip = if s.image_type == ImageType::Photo { help::DETAIL } else { help::QUALITY };
    for q in [Quality::High, Quality::Medium, Quality::Low] {
        // The classifier recommends a (type, quality) pair.
        let rec = class.image_type == s.image_type && class.quality == q;
        let r = option_row(ui, s.quality == q, quality_name(q), None, rec, |_| {});
        if r.on_hover_text(tip).clicked() {
            s.quality = q;
        }
    }
}

fn finder(ui: &mut Ui, f: &mut FinderSettings) {
    ui.label("Colour sensitivity").on_hover_text(help::SENSITIVITY);
    segmented(ui, &mut f.sensitivity, &[(0, "High"), (1, "Medium"), (2, "Low")]).on_hover_text(help::SENSITIVITY);
    ui.checkbox(&mut f.anti_aliased, "Anti-aliased edges").on_hover_text(help::PALETTE_AA);
}

fn colours(ui: &mut Ui, s: &mut Settings, pals: Option<&Palettes>, hover: &mut Option<Hover>) {
    let t = theme::tokens(ui.ctx());
    ui.spacing_mut().item_spacing.y = 4.0;
    let rec = pals.and_then(|p| p.recommended());
    match pals {
        None => {
            ui.horizontal(|ui| {
                ui.add(egui::Spinner::new().size(14.0).color(t.accent));
                ui.label(RichText::new("Analyzing Image Colors…").color(t.muted));
            });
        }
        Some(p) => {
            for (i, pal) in p.found.iter().enumerate() {
                let k = i + 2;
                let selected = match s.palette {
                    PaletteChoice::Recommended => rec == Some(i),
                    PaletteChoice::Count(c) => c == k,
                    _ => false,
                };
                let title = if k == 2 { "Two colours".to_string() } else { format!("{k} colours") };
                let r = option_row(ui, selected, &title, None, rec == Some(i), |ui| {
                    swatches(ui, &pal.colors, 14.0);
                });
                let r = r.on_hover_text(format!("{}\nCost {:.3}", help::FIXED, pal.cost));
                if r.clicked() {
                    s.palette = if rec == Some(i) { PaletteChoice::Recommended } else { PaletteChoice::Count(k) };
                }
                if r.hovered() {
                    *hover = Some(Hover::Palette(pal.colors.clone()));
                }
            }
        }
    }
    let flexible = match s.palette {
        PaletteChoice::Flexible => true,
        PaletteChoice::Recommended => pals.is_some() && rec.is_none(),
        _ => false,
    };
    let r = option_row(
        ui,
        flexible,
        "Flexible palette",
        Some("Colours found while segmenting"),
        pals.is_some() && rec.is_none(),
        |_| {},
    );
    if r.on_hover_text(help::FLEXIBLE).clicked() {
        s.palette = if pals.is_some() && rec.is_none() { PaletteChoice::Recommended } else { PaletteChoice::Flexible };
    }
    let custom = matches!(s.palette, PaletteChoice::Custom(_));
    let r = option_row(ui, custom, "Custom palette", Some("Pick the colours yourself"), false, |_| {});
    if r.clicked() && !custom {
        // Start from the palette in use, else the recommended one.
        let seed = match &s.palette {
            PaletteChoice::Count(k) => pals.map(|p| p.found[k - 2].colors.clone()),
            _ => pals.map(|p| p.found[rec.unwrap_or(0)].colors.clone()),
        };
        s.palette = PaletteChoice::Custom(seed.unwrap_or_else(|| vec![[0, 0, 0, 255], [255, 255, 255, 255]]));
    }
    let r = r.on_hover_text(help::CUSTOM);
    if let PaletteChoice::Custom(colors) = &mut s.palette {
        if r.hovered() {
            *hover = Some(Hover::Palette(colors.clone()));
        }
        custom_editor(ui, colors);
    }
}

/// Edit, add and remove colours of a custom palette (at least one colour stays).
fn custom_editor(ui: &mut Ui, colors: &mut Vec<[u8; 4]>) {
    let mut remove = None;
    ui.horizontal_wrapped(|ui| {
        ui.spacing_mut().item_spacing = egui::vec2(4.0, 4.0);
        for (i, c) in colors.iter_mut().enumerate() {
            let mut c32 = Color32::from_rgba_unmultiplied(c[0], c[1], c[2], c[3]);
            let r = egui::color_picker::color_edit_button_srgba(ui, &mut c32, egui::color_picker::Alpha::OnlyBlend);
            if r.changed() {
                *c = c32.to_srgba_unmultiplied();
            }
            if r.secondary_clicked() {
                remove = Some(i);
            }
            r.on_hover_text(format!("{}\nRight-click to remove", vectorizeo_pipeline::hex_rgba(*c)));
        }
        if ui.small_button("+").on_hover_text("Add a colour").clicked() {
            colors.push(colors.last().copied().unwrap_or([128, 128, 128, 255]));
        }
    });
    if let Some(i) = remove
        && colors.len() > 1
    {
        colors.remove(i);
    }
}

/// A labelled slider over 1..=12 with words at both ends.
fn scale(ui: &mut Ui, label: &str, tip: &str, value: &mut i32, ends: (&str, &str)) {
    let t = theme::tokens(ui.ctx());
    ui.label(label).on_hover_text(tip);
    ui.horizontal(|ui| {
        ui.label(RichText::new(ends.0).size(11.0).color(t.muted));
        ui.spacing_mut().slider_width = (ui.available_width() - 80.0).max(60.0);
        ui.add(Slider::new(value, 1..=12).show_value(false)).on_hover_text(format!("{value} of 12"));
        ui.label(RichText::new(ends.1).size(11.0).color(t.muted));
    });
}

fn segmentation(ui: &mut Ui, a: &mut AdvancedSettings) {
    scale(ui, "Complexity", help::COMPLEXITY, &mut a.segmentation_complexity, ("Fewer", "More"));
    ui.horizontal(|ui| {
        ui.label("Minimum segment size").on_hover_text(help::MIN_PIXELS);
        ui.add(egui::DragValue::new(&mut a.min_num_pixels).range(0..=1000).suffix(" px"));
    });
    ui.label("Anti-aliasing artifact rejection").on_hover_text(help::AA_REJECTION);
    segmented(
        ui,
        &mut a.aa_rejection,
        &[(AaRejection::Level0, "Level 0"), (AaRejection::Level1, "Level 1"), (AaRejection::Level2, "Level 2")],
    )
    .on_hover_text(help::AA_REJECTION);
    ui.checkbox(&mut a.cluster_colors, "Cluster similar colours").on_hover_text(help::CLUSTER);
}

fn smoothing(ui: &mut Ui, a: &mut AdvancedSettings) {
    scale(ui, "Smoothness", help::SMOOTHNESS, &mut a.smoothness, ("Less", "More"));
    ui.checkbox(&mut a.detect_sharp_corners, "Detect sharp corners").on_hover_text(help::CORNERS);
    ui.checkbox(&mut a.anti_aliasing_engine, "Anti-aliasing engine").on_hover_text(help::AA_ENGINE);
    scale(ui, "Number of nodes", help::NODES, &mut a.num_nodes, ("Fewer", "More"));
}

fn export(ui: &mut Ui, s: &mut Settings) {
    ui.spacing_mut().item_spacing.y = 4.0;
    for (mode, title, tip) in [
        (ShapeMode::Stacked, "Stacked", help::STACKED),
        (ShapeMode::CutOuts, "Cut-outs", help::CUTOUTS),
        (ShapeMode::CutOutsGrouped, "Cut-outs, grouped by colour", help::GROUPED),
    ] {
        if option_row(ui, s.shape_mode == mode, title, None, false, |_| {}).on_hover_text(tip).clicked() {
            s.shape_mode = mode;
        }
    }
    ui.add_space(2.0);
    ui.checkbox(&mut s.stroke, "Hairline stroke").on_hover_text(help::STROKE);
}
