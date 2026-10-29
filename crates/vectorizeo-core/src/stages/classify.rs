//! Image classification: loads the `pass1` config, flattens the image, segments it (flexible
//! palette), computes 20 image features and runs a learned decision
//! tree that names the image type and detail level, e.g. `"logo_aa 1"`.
//!
//! The classifier is built on a 16-bin `ColorHistogram`.
//!
//! Quirk: pixels are matched to the histogram peaks with the peak colours left in byte
//! units (0..255) while the pixel is in 0..1 units.

use std::collections::HashSet;

use super::labels::INV_255_F32;
use super::palette::{
    FoundPalette, Histogram, PaletteParams, boundary_band, edge_image, edge_weight, find_palettes_with_peaks,
};
use super::preprocess::shrink_for_analysis;
use super::segmenter::{Segmentation, segment_pixels};
use super::shadow::edge_planes;
use crate::image::RgbaImage;
use crate::params::{EngineConfig, ImageType, Quality};

pub const NUM_FEATURES: usize = 20;

/// A decision-tree leaf name. (An alias so that serde does not borrow it from the input.)
pub type Label = &'static str;

/// Result of 's first half (maps the tree's label).
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Classification {
    /// Tree leaf, e.g. `"photo 1"`.
    #[cfg_attr(feature = "serde", serde(with = "crate::serde_util::static_str"))]
    pub label: Label,
    pub image_type: ImageType,
    pub quality: Quality,
    pub features: [f32; NUM_FEATURES],
}

/// Classify a flattened image. `extension` is the input file's extension with the dot
///one branch of the tree tests it.
pub fn classify(img: &RgbaImage, extension: &str) -> Classification {
    let cfg = EngineConfig::pass1();
    let seg = segment_pixels(img, &cfg);
    let features = image_features(img, &seg, &cfg);
    let label = type_tree(&features, extension);
    let (kind, level) = label.split_once(' ').expect("tree labels are `kind level`");
    // maps: logo → 2, logo_aa → 1, photo → 0; level N → quality index 2 − N.
    let image_type = match kind {
        "photo" => ImageType::Photo,
        "logo_aa" => ImageType::ArtworkBlended,
        _ => ImageType::ArtworkUnblended,
    };
    let quality = match level {
        "2" => Quality::High,
        "1" => Quality::Medium,
        _ => Quality::Low,
    };
    Classification { label, image_type, quality, features }
}

/// The 20 features + 4·i).
pub fn image_features(img: &RgbaImage, seg: &Segmentation, cfg: &EngineConfig) -> [f32; NUM_FEATURES] {
    let (w, h) = (img.width, img.height);
    let n = w * h;
    let px: Vec<[u8; 4]> = img.pixels.iter().map(|c| [c.b, c.g, c.r, c.a]).collect();
    let unit = |c: [u8; 4]| c.map(|v| f32::from(v) * INV_255_F32);
    // planes, band, edge image.
    let (plane_a, plane_b) = edge_planes(img);
    let labels: Vec<usize> = seg.labels.iter().map(|&l| l as usize).collect();
    let band = boundary_band(&labels, w, h, 2);
    let edges = edge_image(&px, w, h);
    let weight = |i: usize| edge_weight(edges[i], cfg.edge_weight_scale, cfg.edge_weight_floor);

    // Edge-weighted histogram and its peaks; how much mass is left once the peaks, their
    // neighbourhoods, and the blend lines between peaks are removed.
    let mut hist = Histogram::new(16);
    for (i, &c) in px.iter().enumerate() {
        hist.add(c, weight(i));
    }
    let mut peaks = hist.peaks(true);
    let t0 = hist.total();
    for &(c, _) in &peaks {
        hist.zero(c, false);
    }
    let t1 = hist.total();
    let narrow = hist.clone();
    for &(c, _) in &peaks {
        hist.zero(c, true);
    }
    let t2 = hist.total();
    hist = narrow;
    for i in 0..peaks.len().saturating_sub(1) {
        let ci = unit(peaks[i].0);
        for j in i + 1..peaks.len() {
            let cj = unit(peaks[j].0);
            let d = [0, 1, 2, 3].map(|c| cj[c] - ci[c]);
            let len = d.iter().fold(0.0f64, |acc, &v| acc + f64::from(v) * f64::from(v)).sqrt();
            // fistp with the rounding mode set to chop.
            let steps = 3 - (len * -16.0) as i64;
            let denom = (steps - 1) as f32;
            for s in 0..steps {
                let t = s as f32 / denom;
                let p = [0, 1, 2, 3].map(|c| d[c] * t + ci[c]);
                hist.zero(p.map(|v| (f64::from(v) * 255.0 + 0.5) as i64 as u8), false);
            }
        }
    }
    let t3 = hist.total();
    if peaks.is_empty() {
        peaks.push(([0; 4], n as f64));
    }
    let kp = peaks.len().min(12);

    // Per pixel: nearest peak (byte-unit peaks, see module docs), weighted colour sums per peak,
    // plane energies, residual to the pass-1 segment mean inside / outside the boundary band.
    let mut map = vec![0usize; n];
    let mut sums = vec![[0.0f32; 4]; kp];
    let mut wsum = vec![0.0f32; kp];
    let mut wtot = 0.0f32;
    let (mut fa, mut fb) = (0.0f32, 0.0f32);
    let mut colours = HashSet::new();
    let (mut ic, mut bc) = (0i32, 0i32);
    let (mut ir2, mut br2, mut ar2) = (0.0f32, 0.0f32, 0.0f32);
    for i in 0..n {
        let p = unit(px[i]);
        let dist = |q: [u8; 4]| {
            let d = [0, 1, 2, 3].map(|c| f64::from(p[c] - f32::from(q[c])));
            ((d[0] * d[0] + d[1] * d[1]) + d[2] * d[2]) + d[3] * d[3]
        };
        let mut best = dist(peaks[0].0);
        let mut bi = 0;
        for (j, peak) in peaks.iter().enumerate().take(kp).skip(1) {
            let dj = dist(peak.0);
            if best > dj {
                best = dj;
                bi = j;
            }
        }
        map[i] = bi;
        let wt = weight(i);
        for c in 0..4 {
            sums[bi][c] += (f64::from(p[c]) * wt) as f32;
        }
        wsum[bi] = (wt + f64::from(wsum[bi])) as f32;
        wtot = (wt + f64::from(wtot)) as f32;
        fa += plane_a[i] * plane_a[i];
        fb += plane_b[i] * plane_b[i];
        // Σ·(1/n); sums are kept in RGBA order.
        let s = &seg.segments[labels[i]];
        let inv = 1.0f32 / s.count as f32;
        let m = [s.sum[2] * inv, s.sum[1] * inv, s.sum[0] * inv, s.sum[3] * inv];
        let e = [0, 1, 2, 3].map(|c| p[c] - m[c]);
        let r2 = ((e[0] * e[0] + e[1] * e[1]) + e[2] * e[2]) + e[3] * e[3];
        colours.insert(px[i]);
        if band[i] == 0 {
            ic += 1;
            ir2 += r2;
        } else {
            bc += 1;
            br2 += r2;
        }
        ar2 += r2;
    }

    // Weighted second and fourth moments of the summed channel residual to the peak means.
    let means: Vec<[f32; 4]> = (0..kp)
        .map(|j| {
            let inv = 1.0f32 / wsum[j];
            sums[j].map(|v| v * inv)
        })
        .collect();
    let (mut m2, mut m4) = (1e-6f32, 0.0f32);
    for i in 0..n {
        let p = unit(px[i]);
        let wt = weight(i) as f32;
        let m = means[map[i]];
        let s = ((p[0] - m[0] + (p[1] - m[1])) + (p[2] - m[2])) + (p[3] - m[3]);
        let s2 = s * s;
        let s4 = s2 * s2;
        m2 += s2 * wt;
        m4 += s4 * wt;
    }
    let inv = 1.0f32 / wtot;
    m4 *= inv;
    m2 *= inv;
    let kurtosis = (f64::from(m4) / (f64::from(m2) * f64::from(m2)) - 3.0) as f32;

    let mut f = [0.0f32; NUM_FEATURES];
    let nf = n as f32;
    let im = if ic != 0 { ir2 / ic as f32 } else { ir2 };
    let bm = if bc != 0 { br2 / bc as f32 } else { br2 };
    let am = if ic + bc != 0 { ar2 / (ic + bc) as f32 } else { ar2 };
    let s = bm + im;
    f[5] = if s > 0.0 { bm / s } else { 0.0 };
    f[6] = am;
    f[7] = bm;
    f[8] = im;
    f[0] = f64::from(fa / nf).sqrt() as f32;
    f[1] = f64::from(fb / nf).sqrt() as f32;
    f[9] = nf;
    let nseg = seg.segments.len();
    f[10] = nseg as f32;
    f[11] = nseg as f32 / nf;
    f[12] = (nseg as f64 / (n as f64).sqrt()) as f32;
    let o = edge_profiles(img);
    f[2] = (o[4] + o[0]) * 0.5;
    f[3] = (o[3] + o[1]) * 0.5;
    f[4] = o[2];
    f[13] = colours.len() as f32;
    f[14] = kurtosis;
    f[15] = m2;
    f[16] = m4;
    let inv0 = 1.0 / t0;
    f[17] = (t1 * inv0) as f32;
    f[18] = (t2 * inv0) as f32;
    f[19] = (t3 * inv0) as f32;
    f
}

/// Mean normalised luma-step profile over every pixel (2 ≤ x < W−3,
/// 2 ≤ y < H−3), horizontal then vertical, where the centre step dominates.
fn edge_profiles(img: &RgbaImage) -> [f32; 5] {
    let (w, h) = (img.width as i64, img.height as i64);
    let mut o = [0.0f32; 5];
    let mut count = 0i32;
    if h - 3 > 2 {
        for y in 2..h - 3 {
            for x in 2..w - 3 {
                for horizontal in [true, false] {
                    if let Some(d) = step_profile(img, x, y, horizontal) {
                        count += 1;
                        for k in 0..5 {
                            o[k] += d[k];
                        }
                    }
                }
            }
        }
        if count != 0 {
            let inv = 1.0f32 / count as f32;
            o = o.map(|v| inv * v);
        }
    }
    o
}

/// Luma (0.3 R + 0.59 G + 0.11 B) of the six pixels at
/// offsets −2..3, the five steps between them oriented so the centre one rises, outer steps
/// against it cleared; accepted if the centre step is the largest and above 0.1. Returns the
/// steps normalised to sum 1.
fn step_profile(img: &RgbaImage, x: i64, y: i64, horizontal: bool) -> Option<[f32; 5]> {
    let luma = |k: i64| {
        let (px, py) = if horizontal { (x + k, y) } else { (x, y + k) };
        let c = img.pixels[py as usize * img.width + px as usize];
        let [r, g, b] = [c.r, c.g, c.b].map(|v| f64::from(f32::from(v) * INV_255_F32));
        (g * 0.59 + r * 0.3) + b * 0.11
    };
    let l: Vec<f64> = (-2..4).map(luma).collect();
    let mut d = [0, 1, 2, 3, 4].map(|k| (l[k + 1] - l[k]) as f32);
    if 0.0 > d[2] {
        d = d.map(|v| 0.0 - v);
    }
    if 0.0 > d[3] {
        d[3] = 0.0;
        d[4] = 0.0;
    } else if 0.0 > d[4] {
        d[4] = 0.0;
    }
    if 0.0 > d[1] {
        d[1] = 0.0;
        d[0] = 0.0;
    } else if 0.0 > d[0] {
        d[0] = 0.0;
    }
    let mut m = d[0];
    for &v in &d[1..] {
        if v > m {
            m = v;
        }
    }
    let ok = d[2] > 0.1f32 && m == d[2];
    let inv = 1.0f32 / ((((d[0] + d[1]) + d[2]) + d[3]) + d[4]);
    let d = d.map(|v| inv * v);
    ok.then_some(d)
}

/// Comparisons are written `!(f < c)`, so that NaN takes the "f ≥ c" branch.
fn type_tree(f: &[f32; NUM_FEATURES], ext: &str) -> &'static str {
    let ge = |i: usize, c: f64| !(f64::from(f[i]) < c);
    if ge(18, 0.0134977) {
        if ge(5, 0.867741) {
            if ge(2, 0.0124402) {
                if ge(0, 0.241446) {
                    if ge(3, 0.132534) { "logo_aa 2" } else { "photo 1" }
                } else if ge(3, 0.183605) {
                    if ge(7, 0.00268996) { if ge(15, 1.19829) { "photo 1" } else { "logo_aa 1" } } else { "photo 1" }
                } else if ge(2, 0.0225216) {
                    if ge(2, 0.103327) { "logo_aa 1" } else { "photo 1" }
                } else if f64::from(f[5]) < 0.915114 {
                    "photo 1"
                } else if f64::from(f[4]) < 0.649337 || ge(13, 80543.5) {
                    "logo_aa 2"
                } else {
                    "photo 1"
                }
            } else if ge(4, 0.984352) {
                "logo 2"
            } else if ge(3, 0.091711) || f64::from(f[7]) < 0.00301763 {
                "logo_aa 2"
            } else {
                "photo 1"
            }
        } else if ge(18, 0.0649332) {
            if ge(0, 0.20806) {
                if ge(0, 0.928628) { "logo_aa 0" } else { "photo 1" }
            } else if ge(15, 0.158062) || ge(4, 0.619184) || ge(7, 0.00190362) {
                "photo 1"
            } else if f64::from(f[1]) < 2.355e-05 {
                "logo_aa 2"
            } else {
                "photo 1"
            }
        } else if f64::from(f[15]) < 0.0323685 {
            "logo_aa 2"
        } else if ge(10, 9497.5) {
            "logo_aa 1"
        } else {
            "photo 1"
        }
    } else if ge(4, 0.960049) {
        if ge(18, 5.16e-07) {
            if ge(0, 0.380912) || !(f64::from(f[2]) < 0.00114401) { "logo_aa 1" } else { "logo_aa 2" }
        } else if ge(2, 0.00130897) {
            "logo_aa 2"
        } else {
            "logo 2"
        }
    } else if ge(2, 0.0197072) {
        if ge(18, 0.00291393) {
            if ge(12, 2.63168) {
                if ge(2, 0.0380126) || ge(0, 0.400423) || ge(3, 0.136656) {
                    "logo_aa 1"
                } else if ge(8, 0.000193388) {
                    "logo_aa 2"
                } else {
                    "photo 1"
                }
            } else if ge(2, 0.0540923) {
                "logo_aa 1"
            } else {
                "photo 1"
            }
        } else if ge(2, 0.0285461) {
            "logo_aa 1"
        } else if ge(4, 0.674069) {
            if ge(18, 0.00132773) {
                "logo_aa 2"
            } else if [".bmp", ".jpeg", ".png", ".tiff"].contains(&ext) {
                "logo_aa 1"
            } else {
                "logo_aa 2"
            }
        } else if f64::from(f[2]) < 0.0238979 {
            "logo_aa 2"
        } else {
            "logo_aa 1"
        }
    } else if ge(18, 0.00219943) {
        if ge(0, 0.406237) || f64::from(f[2]) < 0.0162755 { "logo_aa 2" } else { "photo 1" }
    } else if ge(3, 0.0923396) {
        if ge(2, 0.0111265) {
            if ge(8, 0.000665414) {
                "logo_aa 1"
            } else if 400368.0f32 > f[9] || ge(19, 3.515e-07) {
                // `comiss` against the f32 constant.
                "logo_aa 2"
            } else {
                "logo_aa 1"
            }
        } else {
            "logo_aa 2"
        }
    } else if ge(16, 0.111529) {
        if ge(11, 0.00792502) {
            "logo_aa 2"
        } else if ge(13, 57.5) {
            "logo_aa 1"
        } else {
            "logo_aa 0"
        }
    } else {
        "logo_aa 2"
    }
}

/// Palette mode suggested by the colour-count tree (map).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum PaletteMode {
    /// "2": two colours.
    TwoColors,
    /// "3".."12": fixed palette.
    Fixed,
    /// "13": flexible / unlimited.
    Flexible,
}

/// Colour-count recommendation (second half of).
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ColorRecommendation {
    /// Tree leaf, `"2"`..`"13"`.
    #[cfg_attr(feature = "serde", serde(with = "crate::serde_util::static_str"))]
    pub label: Label,
    pub mode: PaletteMode,
    /// Recommended colour count (map; "13" → 12).
    pub colors: usize,
    #[cfg_attr(feature = "serde", serde(with = "crate::serde_util::big_array"))]
    pub features: [f64; NUM_COLOR_FEATURES],
}

pub const NUM_COLOR_FEATURES: usize = 62;

/// Palettes for k = 2..=12 with the `pass1` config, anti-aliased unless the
/// image was classified as aliased artwork, on the shrunk copy, then features and tree.
pub fn recommend_colors(img: &RgbaImage, image_type: ImageType) -> ColorRecommendation {
    let aa = image_type != ImageType::ArtworkUnblended;
    let img = &shrink_for_analysis(img, aa);
    let params = PaletteParams::new(&EngineConfig::pass1(), aa);
    let (pals, peak_counts) = find_palettes_with_peaks(img, &params, 2, 12);
    recommend_from_palettes(img, &pals, &peak_counts)
}

/// Features and tree over palettes for k = 2..=12, as the palette search
/// does after a 2..=12 run.
pub fn recommend_from_palettes(img: &RgbaImage, pals: &[FoundPalette], peak_counts: &[f64]) -> ColorRecommendation {
    let features = color_features(img, pals, peak_counts);
    let label = color_tree(&features);
    let colors: usize = label.parse().expect("numeric label");
    let mode = match colors {
        2 => PaletteMode::TwoColors,
        13 => PaletteMode::Flexible,
        _ => PaletteMode::Fixed,
    };
    ColorRecommendation { label, mode, colors: colors.min(12), features }
}

/// Colour-count features over the palettes for k = 2..=12
/// (`pals[k − 2]`), as doubles.
pub fn color_features(img: &RgbaImage, pals: &[FoundPalette], peak_counts: &[f64]) -> [f64; NUM_COLOR_FEATURES] {
    assert_eq!(pals.len(), 11, "palettes for k = 2..=12");
    let mut g = [0.0f64; NUM_COLOR_FEATURES];
    let colours: HashSet<[u8; 4]> = img.pixels.iter().map(|c| [c.b, c.g, c.r, c.a]).collect();
    g[0] = colours.len() as f64;
    g[1] = (img.width * img.height) as f64;
    // 13-entry arrays of (value, k); entries 0 and 1 stay (1e6, −1). qsort with
    // orders by value, then k: a strict order, so the sort algorithm does not matter.
    let mut by_cost: Vec<(f64, i32)> = vec![(1e6, -1); 13];
    let mut by_fit: Vec<(f64, i32)> = vec![(1e6, -1); 13];
    let mut fit = [1.0f64; 13];
    let mut min_fit = f64::from(pals[0].fit_cost);
    let mut max_fit = min_fit;
    for k in 2..=12usize {
        let pal = &pals[k - 2];
        let f = f64::from(pal.fit_cost);
        by_cost[k] = (f64::from(pal.cost), k as i32);
        by_fit[k] = (f, k as i32);
        fit[k] = f;
        if min_fit > f {
            min_fit = f;
        }
        if f > max_fit {
            max_fit = f;
        }
    }
    let range = max_fit - min_fit;
    for v in &mut fit[2..=12] {
        *v = (*v - min_fit) / range;
    }
    let order =
        |a: &(f64, i32), b: &(f64, i32)| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal).then(a.1.cmp(&b.1));
    by_cost.sort_by(order);
    by_fit.sort_by(order);
    let mut rank_cost = [-1i32; 13];
    let mut rank_fit = [-1i32; 13];
    for i in 0..13 {
        if by_cost[i].1 >= 2 {
            rank_cost[by_cost[i].1 as usize] = i as i32;
        }
        if by_fit[i].1 >= 2 {
            rank_fit[by_fit[i].1 as usize] = i as i32;
        }
    }
    for i in 0..4 {
        g[2 + i] = f64::from(by_cost[i].1);
        g[6 + i] = f64::from(by_fit[i].1);
        g[10 + i] = by_fit[i].0;
    }
    for k in 2..=12 {
        g[12 + k] = f64::from(rank_cost[k]);
        g[23 + k] = f64::from(rank_fit[k]);
        g[36 + k] = fit[k];
    }
    g[36] = peak_counts.len() as f64;
    // The minimum of an empty array is taken as 0.
    g[37] = array_min(peak_counts);
    g[49] = min_fit;
    g[50] = max_fit;
    for k in 2..=12 {
        g[49 + k] = array_min(&pals[k - 2].counts);
    }
    g
}

/// Smallest element (strict `<`, first element first).
fn array_min(v: &[f64]) -> f64 {
    let Some(&first) = v.first() else { return 0.0 };
    v.iter().fold(first, |m, &x| if x < m { x } else { m })
}

/// Colour-count decision tree, written like [`type_tree`]. Splits that lead to the same leaf
/// either way are folded; the type-label test it contains (`"logo_aa 0/1/2"`) also ends in "13"
/// on every path.
fn color_tree(g: &[f64; NUM_COLOR_FEATURES]) -> &'static str {
    let ge = |i: usize, c: f64| !(g[i] < c);
    let lt = |i: usize, c: f64| g[i] < c;
    if ge(50, 2.69034) {
        if ge(40, 0.288933) {
            if ge(13, 2.04511) {
                if ge(36, 7.5) {
                    "13"
                } else if ge(50, 3.56775) {
                    if lt(41, 0.160751) { "5" } else { "13" }
                } else if ge(51, 3296.27) {
                    "2"
                } else {
                    "13"
                }
            } else if ge(15, 7.5) {
                if ge(36, 5.5) {
                    if ge(3, 8.5) {
                        if lt(44, 0.129347) { "8" } else { "9" }
                    } else if lt(16, 6.5) {
                        "13"
                    } else if ge(42, 0.313304) {
                        "7"
                    } else {
                        "6"
                    }
                } else if ge(41, 0.136787) {
                    "4"
                } else {
                    "5"
                }
            } else if ge(51, 2004.84) {
                if ge(0, 986.0) { "3" } else { "2" }
            } else if ge(36, 6.5) {
                if ge(10, 1.24186) { "13" } else { "8" }
            } else {
                "13"
            }
        } else if ge(39, 0.333629) {
            if ge(15, 6.5) {
                "4"
            } else if ge(12, 2.2514) {
                "13"
            } else if lt(36, 3.5) {
                "3"
            } else if ge(39, 0.413465) {
                "4"
            } else {
                "3"
            }
        } else {
            "3"
        }
    } else if ge(50, 1.73555) {
        if ge(51, 995.704) {
            "2"
        } else if ge(36, 2.5) {
            if ge(39, 0.351533) {
                if ge(13, 0.938407) {
                    "13"
                } else if ge(50, 2.52056) {
                    "4"
                } else {
                    "13"
                }
            } else {
                "3"
            }
        } else {
            "2"
        }
    } else if ge(51, 217.677) || ge(37, 35.25) {
        "2"
    } else {
        "3"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::image::Rgba;

    #[test]
    fn two_colour_logo_gets_a_small_palette() {
        let (w, h) = (64usize, 48usize);
        let img = RgbaImage::new(
            w,
            h,
            (0..w * h)
                .map(|i| {
                    if i % w < 20 {
                        Rgba { r: 30, g: 60, b: 200, a: 255 }
                    } else {
                        Rgba { r: 255, g: 255, b: 255, a: 255 }
                    }
                })
                .collect(),
        );
        let r = recommend_colors(&img, ImageType::ArtworkUnblended);
        assert_eq!(r.features[0], 2.0);
        assert_eq!(r.features[36], 2.0, "two histogram peaks");
        // k = 2 fits exactly, so it has the lowest cost and the best fit.
        assert_eq!((r.features[2], r.features[6]), (2.0, 2.0));
        assert!(r.colors <= 3, "{:?} {:?}", r.label, &r.features[..]);
    }

    #[test]
    fn step_profile_finds_a_sharp_edge() {
        let img = RgbaImage::new(
            8,
            1,
            (0..8)
                .map(
                    |x| if x < 4 { Rgba { r: 0, g: 0, b: 0, a: 255 } } else { Rgba { r: 255, g: 255, b: 255, a: 255 } },
                )
                .collect(),
        );
        let d = step_profile(&img, 3, 0, true).expect("centre step dominates");
        assert_eq!(d, [0.0, 0.0, 1.0, 0.0, 0.0]);
        // Falling edges are oriented the same way.
        assert!(step_profile(&img, 2, 0, true).is_none());
    }

    #[test]
    fn flat_logo_is_not_a_photo() {
        let (w, h) = (64usize, 48usize);
        let img = RgbaImage::new(
            w,
            h,
            (0..w * h)
                .map(|i| {
                    let (x, y) = (i % w, i / w);
                    if (x as i32 - 30).pow(2) + (y as i32 - 24).pow(2) < 300 {
                        Rgba { r: 200, g: 20, b: 20, a: 255 }
                    } else {
                        Rgba { r: 255, g: 255, b: 255, a: 255 }
                    }
                })
                .collect(),
        );
        let c = classify(&img, ".png");
        assert_ne!(c.image_type, ImageType::Photo, "{c:?}");
        assert_eq!(c.features[13], 2.0);
    }
}
