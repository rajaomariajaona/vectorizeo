//! Preprocessor: `{filter_type, reduce_noise_order}`.
//!
//! Runs once per segmentation on the flattened, premultiplied image, before it becomes the
//! working buffer:
//! - filter_type 1: `enhance`, `reduce_noise_order` times;
//! - filter_type 2: noise reduction (not supported);
//! - filter_type 3: despeckle (not supported).
//!
//! Every preset uses type 1 (order 0–3); types 2 and 3 are not supported.

use crate::image::{Rgba, RgbaImage};
use crate::params::EngineConfig;

/// with the config's `filter_type` / `reduce_noise_order`.
pub fn preprocess(img: &RgbaImage, cfg: &EngineConfig) -> RgbaImage {
    match cfg.filter_type {
        1 => {
            let mut out = img.clone();
            for _ in 0..cfg.reduce_noise_order {
                out = enhance(&out);
            }
            out
        }
        2 => unimplemented!("Preprocessor filter_type 2 (ReduceNoiseImage)"),
        3 => unimplemented!("Preprocessor filter_type 3 (DespeckleImage)"),
        _ => img.clone(),
    }
}

/// `enhance` weights, row by row.
const WEIGHTS: [[f64; 5]; 5] = [
    [5.0, 8.0, 10.0, 8.0, 5.0],
    [8.0, 20.0, 40.0, 20.0, 8.0],
    [10.0, 40.0, 80.0, 40.0, 10.0],
    [8.0, 20.0, 40.0, 20.0, 8.0],
    [5.0, 8.0, 10.0, 8.0, 5.0],
];

/// 1/255.
const INV_255: f64 = 0.00392156862745098;

/// Edge-preserving enhancement (8-bit quantum): each pixel becomes the
/// weighted mean of the 5×5 neighbours whose colour distance to it is below 255²/25,
/// `trunc((Σw·c + W/2 − 1)·(1/W))` per channel. Channels are red, green, blue and opacity
/// (255 − alpha); the border is extended (edge virtual pixels). Images under 5×5 are returned
/// unchanged. Arithmetic is f64; the aggregates are f32 but stay exact.
pub fn enhance(img: &RgbaImage) -> RgbaImage {
    let (w, h) = (img.width, img.height);
    if w < 5 || h < 5 {
        return img.clone();
    }
    let px: Vec<[f64; 4]> = img.pixels.iter().map(|c| [c.r, c.g, c.b, 255 - c.a].map(f64::from)).collect();
    let at = |x: i64, y: i64| px[y.clamp(0, h as i64 - 1) as usize * w + x.clamp(0, w as i64 - 1) as usize];
    let mut out = Vec::with_capacity(w * h);
    for y in 0..h as i64 {
        for x in 0..w as i64 {
            let c = at(x, y);
            let mut agg = [0.0f32; 4];
            let mut total = 0.0f64;
            for (j, row) in WEIGHTS.iter().enumerate() {
                for (i, &wt) in row.iter().enumerate() {
                    let r = at(x + i as i64 - 2, y + j as i64 - 2);
                    let d = [0, 1, 2, 3].map(|k| r[k] - c[k]);
                    let mut dist = ((r[0] + c[0]) * 0.5 + 512.0) * d[0] * d[0] * INV_255;
                    dist += d[1] * d[1] * 4.0;
                    dist += (767.0 - (c[2] + r[2]) * 0.5) * d[2] * d[2] * INV_255;
                    dist += (767.0 - (c[3] + r[3]) * 0.5) * d[3] * d[3] * INV_255;
                    if dist < 2601.0 {
                        for k in 0..4 {
                            agg[k] = (r[k] * wt + f64::from(agg[k])) as f32;
                        }
                        total += wt;
                    }
                }
            }
            let half = f64::from((total * 0.5) as f32);
            let inv = 1.0 / total;
            let q = agg.map(|a| ((half + f64::from(a) - 1.0) * inv) as i32 as u8);
            out.push(Rgba { r: q[0], g: q[1], b: q[2], a: 255 - q[3] });
        }
    }
    RgbaImage::new(w, h, out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn image(w: usize, h: usize, f: impl Fn(usize, usize) -> Rgba) -> RgbaImage {
        RgbaImage::new(w, h, (0..w * h).map(|i| f(i % w, i / w)).collect())
    }

    #[test]
    fn flat_image_is_unchanged() {
        let c = Rgba { r: 200, g: 10, b: 77, a: 255 };
        let img = image(8, 6, |_, _| c);
        assert!(enhance(&img).pixels.iter().all(|&p| p == c));
    }

    #[test]
    fn noise_is_averaged_but_edges_are_kept() {
        // Left half dark, right half light; one slightly brighter pixel in the dark half.
        let img = image(12, 8, |x, y| {
            let v = if x < 6 { 40 } else { 220 };
            let v = if (x, y) == (2, 3) { 52 } else { v };
            Rgba { r: v, g: v, b: v, a: 255 }
        });
        let out = enhance(&img);
        // The bump is pulled towards its neighbours: (52·80 + 40·(W−80) + W/2 − 1) / W.
        let p = out.pixels[3 * 12 + 2];
        assert!(p.r > 40 && p.r < 52, "{p:?}");
        // The step (distance ≫ 51) never mixes across the edge.
        assert_eq!(out.pixels[3 * 12 + 5].r, 40);
        assert_eq!(out.pixels[3 * 12 + 6].r, 220);
    }

    #[test]
    fn large_images_shrink_to_about_150k_pixels() {
        let (w, h) = (800usize, 400usize);
        let img = image(w, h, |x, _| {
            if x < 400 { Rgba { r: 200, g: 0, b: 0, a: 255 } } else { Rgba { r: 0, g: 0, b: 255, a: 255 } }
        });
        for aa in [false, true] {
            let s = shrink_for_analysis(&img, aa);
            // s = √ → 547.7 × 273.9 → 548 × 274.
            assert_eq!((s.width, s.height), (548, 274));
            assert_eq!(s.pixels[0], img.pixels[0]);
            assert_eq!(s.pixels[547], img.pixels[799]);
        }
        let small = image(10, 10, |_, _| Rgba { r: 1, g: 2, b: 3, a: 255 });
        assert_eq!(shrink_for_analysis(&small, true).pixels, small.pixels);
    }

    #[test]
    fn tiny_images_pass_through() {
        let img = image(4, 9, |x, _| Rgba { r: x as u8 * 60, g: 0, b: 0, a: 255 });
        assert_eq!(enhance(&img).pixels, img.pixels);
    }
}

/// Analysis size limit: the palette search and the colour-count classifier work on a copy shrunk
/// to about 150 000 pixels when the image has more than 200 000. Anti-aliased input
/// is resized with the Triangle filter, otherwise point-sampled.
pub fn shrink_for_analysis(img: &RgbaImage, anti_aliased: bool) -> RgbaImage {
    let (w, h) = (img.width, img.height);
    let n = w * h;
    if n <= 200_000 {
        return img.clone();
    }
    let s = (150_000.0 / n as f64).sqrt();
    // fistp with the rounding mode set to chop.
    let gw = (w as f64 * s + 0.5) as usize;
    let gh = (h as f64 * s + 0.5) as usize;
    let (cw, ch) = fit_geometry(w, h, gw, gh);
    if anti_aliased { resize_triangle(img, cw, ch) } else { sample(img, cw, ch) }
}

/// Fit a plain `WxH` geometry: keep the aspect ratio,
/// `scale = min(W/w, f32(H/h))`, sides `max(1, trunc(side·scale + 0.5))`.
fn fit_geometry(w: usize, h: usize, gw: usize, gh: usize) -> (usize, usize) {
    if (gw, gh) == (w, h) || w == 0 || h == 0 {
        return (gw, gh);
    }
    let sw = gw as f64 / w as f64;
    let sh = (gh as f64 / h as f64) as f32;
    let scale = if sw > f64::from(sh) { f64::from(sh) } else { sw };
    let side = |v: usize| ((v as f64 * scale + 0.5) as u32).max(1) as usize;
    (side(w), side(h))
}

/// Point sampling: source column `trunc((x + 0.5)·w/cw)`.
fn sample(img: &RgbaImage, cw: usize, ch: usize) -> RgbaImage {
    let xs: Vec<usize> = (0..cw).map(|x| ((x as f64 + 0.5) * img.width as f64 / cw as f64) as usize).collect();
    let ys: Vec<usize> = (0..ch).map(|y| ((y as f64 + 0.5) * img.height as f64 / ch as f64) as usize).collect();
    let mut px = Vec::with_capacity(cw * ch);
    for &y in &ys {
        for &x in &xs {
            px.push(img.pixels[y * img.width + x]);
        }
    }
    RgbaImage::new(cw, ch, px)
}

/// Triangle-filter resize with blur 1: the two
/// passes (horizontal, vertical), horizontal first iff
/// `cw·(h + ch) > ch·(w + cw)`. Pixels are B, G, R, opacity; weights are f32.
fn resize_triangle(img: &RgbaImage, cw: usize, ch: usize) -> RgbaImage {
    let (w, h) = (img.width, img.height);
    let px: Vec<[u8; 4]> = img.pixels.iter().map(|c| [c.b, c.g, c.r, 255 - c.a]).collect();
    let out = if (cw as u64) * (h as u64 + ch as u64) > (ch as u64) * (w as u64 + cw as u64) {
        let t = filter_pass(&px, w, h, cw, true);
        filter_pass(&t, cw, h, ch, false)
    } else {
        let t = filter_pass(&px, w, h, ch, false);
        filter_pass(&t, w, ch, cw, true)
    };
    RgbaImage::new(cw, ch, out.iter().map(|p| Rgba { r: p[2], g: p[1], b: p[0], a: 255 - p[3] }).collect())
}

/// One pass of `HorizontalFilter` / `VerticalFilter` (image with matte) resizing the `horizontal`
/// axis of a `w`×`h` image to `size`.
fn filter_pass(px: &[[u8; 4]], w: usize, h: usize, size: usize, horizontal: bool) -> Vec<[u8; 4]> {
    const EPSILON: f64 = 1e-6; // minimum weight / filter epsilon
    let (len, other) = if horizontal { (w, h) } else { (h, w) };
    let factor = (size as f64 / len as f64) as f32;
    let mut scale = 1.0 / f64::from(factor);
    if scale <= 1.0 {
        scale = 1.0;
    }
    let mut scale = scale as f32;
    // GetResizeFilterSupport = support (1) · blur (1).
    let mut support = 1.0f32 * scale;
    if f64::from(support) <= 0.5 {
        support = 0.5;
        scale = 1.0;
    }
    let scale = (1.0 / f64::from(scale)) as f32;
    let inv_factor = 1.0f32 / factor;
    let mut out = vec![[0u8; 4]; if horizontal { size * h } else { w * size }];
    for i in 0..size {
        let c = (i as f64 + 0.5) * f64::from(inv_factor);
        let center = c as f32;
        let mut lo = c - f64::from(support) - EPSILON;
        if lo <= 0.0 {
            lo = 0.0;
        }
        let start = (lo + 0.5) as i64;
        let mut hi = f64::from(center) + f64::from(support);
        if !(hi < len as f64) {
            hi = len as f64;
        }
        let stop = (hi + 0.5) as i64;
        let mut weights: Vec<(usize, f32)> = Vec::new();
        let mut density = 0.0f32;
        for p in start..stop {
            let x = (f64::from(scale) * ((p as f64 - f64::from(center)) + 0.5)) as f32;
            let xb = (f64::from(x).abs() / 1.0) as f32;
            let tri = if f64::from(xb) < 1.0 { 1.0 - f64::from(xb) } else { 0.0 };
            let wt = (tri * 1.0) as f32;
            density = (f64::from(density) + f64::from(wt)) as f32;
            weights.push((p as usize, wt));
        }
        if density != 0.0 && density != 1.0 {
            let d = (1.0 / f64::from(density)) as f32;
            for (_, wt) in &mut weights {
                *wt = (f64::from(d) * f64::from(*wt)) as f32;
            }
        }
        for j in 0..other {
            let mut acc = [0.0f32; 4];
            let mut gamma = 0.0f32;
            for &(p, wt) in &weights {
                let q = if horizontal { px[j * w + p] } else { px[p * w + j] };
                let alpha = ((255.0 - f64::from(q[3])) * f64::from(wt) * 0.00392156862745098) as f32;
                for k in 0..3 {
                    acc[k] = (f64::from(q[k]) * f64::from(alpha) + f64::from(acc[k])) as f32;
                }
                acc[3] = (f64::from(q[3]) * f64::from(wt) + f64::from(acc[3])) as f32;
                gamma = (f64::from(gamma) + f64::from(alpha)) as f32;
            }
            let g = 1.0 / if f64::from(gamma).abs() <= EPSILON { 1.0 } else { f64::from(gamma) };
            let mut q = [0u8; 4];
            for k in 0..3 {
                q[k] = round_to_byte(f64::from(acc[k]) * g);
            }
            q[3] = round_to_byte(f64::from(acc[3]));
            let o = if horizontal { j * size + i } else { i * w + j };
            out[o] = q;
        }
    }
    out
}

/// Round to a byte: ≤ 0 → 0, f32 value ≥ 255 → 255, else `trunc(v + 0.5)`.
fn round_to_byte(v: f64) -> u8 {
    if v <= 0.0 {
        0
    } else if !((v as f32) < 255.0) {
        255
    } else {
        (v + 0.5) as u8
    }
}
