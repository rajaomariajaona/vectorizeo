//! Inputs and classifier of the anti-aliasing "shadow" detector.
//!
//! A shadow is a thin segment made of blended pixels between two shapes. The detector computes
//! 42 features per segment; the learned tree flags shadows and
//! decides whether dissolving one into its neighbours is kept. Only the features the trees read
//! are computed (the others feed the debug CSV dump only).

use super::labels::INV_255_F32;
use crate::image::RgbaImage;

/// Number of entries in a feature vector.
pub(crate) const NUM_FEATURES: usize = 42;

fn rgba_f32(img: &RgbaImage, p: usize) -> [f32; 4] {
    let c = img.pixels[p];
    [c.r, c.g, c.b, c.a].map(|v| f32::from(v) * INV_255_F32)
}

/// Neighbour index along one axis: +1, or −1 at the far edge (`(i < n−1)·2 − 1`).
fn fwd(i: usize, n: usize) -> usize {
    if i + 1 < n { i + 1 } else { i.saturating_sub(1) }
}

/// Neighbour index along one axis: −1, or +1 at the near edge (`(i < 1)·2 − 1`).
fn back(i: usize, n: usize) -> usize {
    if i < 1 { (i + 1).min(n - 1) } else { i - 1 }
}

fn dist2(a: [f32; 4], b: [f32; 4]) -> f32 {
    let d = [0, 1, 2, 3].map(|i| a[i] - b[i]);
    d[0] * d[0] + d[1] * d[1] + d[2] * d[2] + d[3] * d[3]
}

/// Plane A: Σ squared RGBA differences to the 4 neighbours (mirrored at the border) of the image.
/// Plane B: squared difference between the blurred image and the mean of its 4 neighbours.
pub(crate) fn edge_planes(img: &RgbaImage) -> (Vec<f32>, Vec<f32>) {
    let (w, h) = (img.width, img.height);
    let blurred = blur(img);
    let mut a = vec![0.0f32; w * h];
    let mut b = vec![0.0f32; w * h];
    for y in 0..h {
        for x in 0..w {
            let p = y * w + x;
            let nbrs = [y * w + fwd(x, w), y * w + back(x, w), fwd(y, h) * w + x, back(y, h) * w + x];
            let c = rgba_f32(img, p);
            let mut s = dist2(c, rgba_f32(img, nbrs[0]));
            s += dist2(c, rgba_f32(img, nbrs[1]));
            s += dist2(c, rgba_f32(img, nbrs[2]));
            a[p] = dist2(c, rgba_f32(img, nbrs[3])) + s;
            let f = |q: usize| blurred[q].map(|v| f32::from(v) * INV_255_F32);
            let (fp, fr, fl, fd, fu) = (f(p), f(nbrs[0]), f(nbrs[1]), f(nbrs[2]), f(nbrs[3]));
            let d = [0, 1, 2, 3].map(|i| fp[i] - (fr[i] + fl[i] + fd[i] + fu[i]) * 0.25);
            b[p] = d[0] * d[0] + d[1] * d[1] + d[2] * d[2] + d[3] * d[3];
        }
    }
    (a, b)
}

/// Gaussian blur (radius 1.5, sigma 0.75) on the RGB channels (not opacity). Separable 5-tap
/// kernel, rows then columns, each pass rounded to 8 bits, edge pixels repeated outside the image.
fn blur(img: &RgbaImage) -> Vec<[u8; 4]> {
    const RADIUS: f64 = 1.5;
    const SIGMA: f64 = 0.75;
    const RANK: i64 = 3;
    let width = (2.0 * RADIUS.ceil() + 1.0) as usize;
    let mut kernel = vec![0.0f64; width];
    let bias = RANK * width as i64 / 2;
    for i in -bias..=bias {
        let alpha = (-((i * i) as f64) / (2.0 * (RANK * RANK) as f64 * SIGMA * SIGMA)).exp();
        kernel[((i + bias) / RANK) as usize] += alpha / ((2.0 * std::f64::consts::PI).sqrt() * SIGMA);
    }
    let norm: f64 = kernel.iter().sum();
    kernel.iter_mut().for_each(|k| *k /= norm);

    let (w, h) = (img.width, img.height);
    let half = (width / 2) as i64;
    let round = |v: f64| {
        if v <= 0.0 {
            0
        } else if v >= 255.0 {
            255
        } else {
            (v + 0.5) as u8
        }
    };
    let src: Vec<[u8; 4]> = img.pixels.iter().map(|c| [c.r, c.g, c.b, c.a]).collect();
    let pass = |src: &[[u8; 4]], horizontal: bool| -> Vec<[u8; 4]> {
        let mut out = src.to_vec();
        for y in 0..h {
            for x in 0..w {
                let mut acc = [0.0f64; 3];
                for (k, &kv) in kernel.iter().enumerate() {
                    let o = k as i64 - half;
                    let q = if horizontal {
                        y * w + (x as i64 + o).clamp(0, w as i64 - 1) as usize
                    } else {
                        (y as i64 + o).clamp(0, h as i64 - 1) as usize * w + x
                    };
                    for ch in 0..3 {
                        acc[ch] += kv * f64::from(src[q][ch]);
                    }
                }
                for ch in 0..3 {
                    out[y * w + x][ch] = round(acc[ch]);
                }
            }
        }
        out
    };
    let rows = pass(&src, true);
    pass(&rows, false)
}

/// Power iteration from (1,1,1,1); returns the normalised
/// vector of the last iteration. The convergence tolerance has no effect.
pub(crate) fn principal_axis(m: &[f32; 16]) -> [f32; 4] {
    let mul = |v: &[f32; 4]| -> [f32; 4] {
        [0, 1, 2, 3].map(|r| m[r * 4] * v[0] + m[r * 4 + 1] * v[1] + m[r * 4 + 2] * v[2] + m[r * 4 + 3] * v[3])
    };
    let mut v = [1.0f32; 4];
    let mut w = mul(&v);
    for _ in 0..10 {
        let len = (w[0] * w[0] + w[1] * w[1] + w[2] * w[2] + w[3] * w[3]).sqrt();
        if len > 0.0 {
            let inv = 1.0 / len;
            w = w.map(|c| inv * c);
        }
        v = w;
        w = mul(&v);
    }
    v
}

/// Luma used by the features. The engine stores pixels as BGRA and weights channel 0 by 0.11,
/// channel 1 by 0.59 and channel 2 by 0.3, so in RGBA order this is 0.3 R + 0.59 G + 0.11 B.
pub(crate) fn luma(c: [f32; 4]) -> f64 {
    f64::from(c[0]) * 0.3 + f64::from(c[1]) * 0.59 + f64::from(c[2]) * 0.11
}

/// `c ≤ f`: also true when `f` is NaN.
fn ge(f: f64, c: f64) -> bool {
    !(c > f)
}

/// `f < c`: false when `f` is NaN.
fn lt(f: f64, c: f64) -> bool {
    c > f
}

/// Learned decision tree; true marks the segment as a shadow.
pub(crate) fn is_shadow(f: &[f64; NUM_FEATURES]) -> bool {
    if ge(f[0x17], 0.00402429) {
        return lt(f[0x17], 0.0241458) && lt(f[5], 0.0473801) && (ge(f[0x16], 0.0138127) || ge(f[0x27], 0.0102382));
    }
    if ge(f[0x12], 1.5) {
        if ge(f[0x14], 0.5) {
            if ge(f[0x13], 0.5) {
                if lt(f[0x29], 0.449972) {
                    return if ge(f[0x15], 0.118876) { lt(f[0xb], 5.5) } else { lt(f[0], 41.535) };
                }
            } else {
                return lt(f[1], 0.0112894);
            }
        }
        return false;
    }
    if ge(f[0xe], 1.5) {
        return if ge(f[5], 0.0491163) { ge(f[0x16], 0.196468) } else { lt(f[0x14], 1.5) };
    }
    false
}

/// Learned decision tree; true keeps a dissolved shadow, false restores it.
/// `mode` is, which nothing in the engine writes; taken as 0.
pub(crate) fn keep_dissolved(f: &[f64; NUM_FEATURES], mode: i32) -> bool {
    if ge(f[0x25], 3.57554) {
        return lt(f[0x23], 0.00258463);
    }
    if ge(f[0x1c], 0.0442212) {
        if mode == 0 {
            return true;
        }
        return lt(f[0x22], 0.694472) && ge(f[0], 3.09081);
    }
    if ge(f[0x27], 0.00627504) || mode == 1 || mode == 2 {
        return true;
    }
    ge(f[4], 0.934367)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Rgba;

    #[test]
    fn blur_keeps_flat_images_flat() {
        let px = Rgba { r: 10, g: 200, b: 77, a: 255 };
        let img = RgbaImage::new(6, 4, vec![px; 24]);
        assert!(blur(&img).iter().all(|&c| c == [10, 200, 77, 255]));
        let (a, b) = edge_planes(&img);
        assert!(a.iter().chain(&b).all(|&v| v == 0.0));
    }

    #[test]
    fn principal_axis_of_a_diagonal_matrix() {
        let mut m = [0.0f32; 16];
        m[0] = 1.0;
        m[5] = 4.0;
        m[10] = 2.0;
        m[15] = 0.5;
        let v = principal_axis(&m);
        assert!(v[1] > 0.99, "{v:?}");
    }
}
