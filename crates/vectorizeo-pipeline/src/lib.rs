//! Pipeline steps shared by the `vectorizeo` commands, the interactive wizard and `vectorizeo-gui`, and
//! the GUI flow's settings and per-stage results ([`session`]).

pub mod edit;
pub mod session;

use std::path::Path;
use std::str::FromStr;

use vectorizeo_core::params::{EngineConfig, ImageType};
use vectorizeo_core::stages::bezier::{BezierPaths, fit_beziers};
use vectorizeo_core::stages::boundary::{BoundaryGraph, trace_boundaries};
use vectorizeo_core::stages::palette::PaletteParams;
use vectorizeo_core::stages::preprocess::shrink_for_analysis;
use vectorizeo_core::stages::segmenter::{Segmentation, segment_pixels, segment_pixels_fixed};
use vectorizeo_core::stages::smoother::{AaInput, smooth};
use vectorizeo_core::{Rgba, RgbaImage};

/// GUI `flatten_image_mode`: 1 white, 2 custom colour, 3 leave transparent.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Flatten(pub Option<[f32; 4]>);

impl Flatten {
    pub const WHITE: Flatten = Flatten(Some([1.0; 4]));

    /// Background colour `RRGGBB` (`#` optional), as QColor::setRedF etc. from the 8-bit picker value.
    pub fn from_hex(s: &str) -> Option<Flatten> {
        let hex = s.strip_prefix('#').unwrap_or(s);
        let v = u32::from_str_radix(hex, 16).ok().filter(|_| hex.len() == 6)?;
        let c = |shift: u32| ((v >> shift) & 0xff) as f32 / 255.0;
        Some(Flatten(Some([c(16), c(8), c(0), 1.0])))
    }
}

/// `none`, `white` or `RRGGBB`.
impl FromStr for Flatten {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, String> {
        match s {
            "none" => Ok(Flatten(None)),
            "white" => Ok(Flatten::WHITE),
            _ => Flatten::from_hex(s).ok_or_else(|| "expected none, white or RRGGBB".into()),
        }
    }
}

/// Comma-separated `RRGGBB` or `RRGGBBAA` colours (`#` optional).
pub fn parse_colors(s: &str) -> Result<Vec<[u8; 4]>, String> {
    s.split(',')
        .map(|c| {
            let hex = c.trim().trim_start_matches('#');
            let v = u32::from_str_radix(hex, 16).map_err(|_| format!("bad colour {c:?}"))?;
            match hex.len() {
                6 => Ok([(v >> 16) as u8, (v >> 8) as u8, v as u8, 255]),
                8 => Ok([(v >> 24) as u8, (v >> 16) as u8, (v >> 8) as u8, v as u8]),
                _ => Err(format!("bad colour {c:?}: expected RRGGBB or RRGGBBAA")),
            }
        })
        .collect()
}

/// Load a bitmap and premultiply it.
pub fn load(path: &Path) -> Result<RgbaImage, String> {
    let img = image::open(path).map_err(|e| format!("cannot open {}: {e}", path.display()))?.to_rgba8();
    Ok(premultiplied(&img))
}

/// A decoded bitmap as the engine loads it: premultiplied.
pub fn premultiplied(img: &image::RgbaImage) -> RgbaImage {
    let (w, h) = img.dimensions();
    let pixels = img.pixels().map(|p| Rgba { r: p[0], g: p[1], b: p[2], a: p[3] }).collect();
    let mut out = RgbaImage::new(w as usize, h as usize, pixels);
    out.premultiply();
    out
}

/// for flatten modes 1 and 2; mode 3 keeps the loaded image.
pub fn flatten(img: &RgbaImage, flatten: Flatten) -> RgbaImage {
    match flatten {
        Flatten(Some(bg)) => img.flatten(bg),
        Flatten(None) => img.clone(),
    }
}

pub fn hex_rgba(c: [u8; 4]) -> String {
    format!("#{:02x}{:02x}{:02x}{:02x}", c[0], c[1], c[2], c[3])
}

/// The palette search works on a copy shrunk to about 150 000 pixels.
pub fn palette_input(img: &RgbaImage, image_type: ImageType) -> RgbaImage {
    shrink_for_analysis(img, image_type == ImageType::ArtworkBlended)
}

/// The palette search parameters: `pass1` config, anti-aliased iff blended artwork.
pub fn palette_params(image_type: ImageType) -> PaletteParams {
    PaletteParams::new(&EngineConfig::pass1(), image_type == ImageType::ArtworkBlended)
}

/// File extension as the GUI stores it for the classifier: lowercase with the dot.
pub fn extension(path: &Path) -> String {
    path.extension().map(|e| format!(".{}", e.to_string_lossy().to_lowercase())).unwrap_or_default()
}

/// Segmentation of the preprocessed image, with a fixed palette or the flexible one.
pub fn segment_with(work: &RgbaImage, cfg: &EngineConfig, palette: Option<&[[u8; 4]]>) -> Segmentation {
    match palette {
        None => segment_pixels(work, cfg),
        Some(palette) => segment_pixels_fixed(work, cfg, palette),
    }
}

/// Output colour of every segment.
pub fn segment_colors(seg: &Segmentation) -> Vec<[u8; 4]> {
    (0..seg.segments.len()).map(|k| seg.segment_color(k)).collect()
}

/// Every pixel in its segment's colour.
pub fn segmentation_image(seg: &Segmentation) -> image::RgbaImage {
    let colors = segment_colors(seg);
    let mut out = image::RgbaImage::new(seg.width as u32, seg.height as u32);
    for (i, px) in out.pixels_mut().enumerate() {
        *px = image::Rgba(colors[seg.labels[i] as usize]);
    }
    out
}

/// The vector result of a segmentation: contours, optional Bezier fits and segment colours.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Vectorized {
    pub graph: BoundaryGraph,
    pub curves: Option<BezierPaths>,
    pub colors: Vec<[u8; 4]>,
}

/// Trace and smooth the contours; returns
/// the graph and the segment colours. `work` is the preprocessed image the segmentation saw.
pub fn trace_smooth(
    seg: &Segmentation,
    work: &RgbaImage,
    cfg: &EngineConfig,
    smooth_contours: bool,
) -> (BoundaryGraph, Vec<[u8; 4]>) {
    let mut graph = trace_boundaries(seg.width, seg.height, &seg.labels, seg.segments.len(), true);
    let colors = segment_colors(seg);
    if smooth_contours {
        let counts: Vec<i32> = seg.segments.iter().map(|s| s.count).collect();
        smooth(&mut graph, &counts, cfg, Some(AaInput { image: work, labels: &seg.labels, colors: colors.clone() }));
    }
    (graph, colors)
}

/// Fit Beziers to the smoothed contours.
pub fn fit(graph: &BoundaryGraph, cfg: &EngineConfig) -> BezierPaths {
    fit_beziers(graph, cfg)
}

/// Trace, smooth and fit Beziers ([`trace_smooth`] then [`fit`]); `cfg` is the smoothing/fitting config.
pub fn vectorize(
    seg: &Segmentation,
    work: &RgbaImage,
    cfg: &EngineConfig,
    smooth_contours: bool,
    fit_curves: bool,
) -> Vectorized {
    let (graph, colors) = trace_smooth(seg, work, cfg, smooth_contours);
    let curves = fit_curves.then(|| fit(&graph, cfg));
    Vectorized { graph, curves, colors }
}

/// Scale at which the wizard previews a result: twice the bitmap size, capped at 1600 px.
pub fn preview_scale(width: usize, height: usize) -> f32 {
    (2.0f32).min(1600.0 / width.max(height).max(1) as f32).max(0.1)
}

/// Rasterise an exported SVG (what will be saved) at `scale` × its size, straight alpha.
pub fn render_svg(svg: &str, scale: f32) -> Option<image::RgbaImage> {
    let tree = resvg::usvg::Tree::from_str(svg, &resvg::usvg::Options::default()).ok()?;
    let size = tree.size();
    let (w, h) = ((size.width() * scale).ceil() as u32, (size.height() * scale).ceil() as u32);
    let mut pixmap = resvg::tiny_skia::Pixmap::new(w.max(1), h.max(1))?;
    resvg::render(&tree, resvg::tiny_skia::Transform::from_scale(scale, scale), &mut pixmap.as_mut());
    let mut out = image::RgbaImage::new(pixmap.width(), pixmap.height());
    for (px, c) in out.pixels_mut().zip(pixmap.pixels()) {
        let c = c.demultiply();
        *px = image::Rgba([c.red(), c.green(), c.blue(), c.alpha()]);
    }
    Some(out)
}
