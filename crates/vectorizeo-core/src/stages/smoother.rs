//! Contour smoother: moves boundary vertices to minimise
//! a data term plus a smoothness prior, using Fletcher–Reeves conjugate gradients with a
//! quadratic-fit line search. Three phases with their own weights and optimizer settings.
//!
//! It consists of:
//! - driver, phase runner and setup, potential and gradient;
//! - priors: type 3 ("sqrt") and type 0 (quadratic);
//! - border snapping and interior clamping after every step;
//! - the optimizer (CG with restarts) and line search type 0;
//! - corner puncturing with a learned corner classifier;
//! - the anti-aliased fit (measurement type 1; geometry in `coverage.rs`), with initial
//!   sub-pixel placement and jitter and the blind-step line search type 5;
//! - anti-inversion (anti-aliased images): a hook after every line search, checks in
//!   `inversion.rs`, and the terms they switch on (vertex flag 0x10 quadratic prior, per-vertex
//!   anchor term) in the potential and gradient.

use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet};

use super::boundary::BoundaryGraph;
use super::coverage::{Piece, piece_gradient, polygon_cells};
use super::inversion::{self, AntiInversion};
use super::labels::{INV_255_F32, Lcg, NBR_DX, NBR_DY};
use super::subpixel::mix;
use crate::image::RgbaImage;
use crate::params::{CgSettings, EngineConfig};

/// Segments with fewer pixels use the quadratic prior and the air-pressure area term.
const SMALL_SEGMENT_PIXELS: i32 = 8;
/// Interior vertices are clamped this far inside the image.
const EDGE_MARGIN: f64 = 0.01;

pub struct ContourSmoother<'a> {
    g: &'a BoundaryGraph,
    /// Pixel count per segment.
    pixel_counts: &'a [i32],
    /// Original positions as f32, the anchor of the measurement term.
    orig: Vec<[f32; 2]>,
    /// Edge entry bit 0: vertex is a punctured corner and excluded from the prior.
    corner: Vec<Vec<bool>>,
    /// Edge entry bit 3: centre turn is the neighbourhood maximum.
    turn_peak: Vec<Vec<bool>>,
    /// Vertices promoted to type 3 by puncturing.
    punctured: Vec<bool>,
    // Per-phase weights.
    strength: f64,
    length_weight: f64,
    air_pressure: f64,
    measurement: i32,
    /// Image, labels and segment colours, needed by the anti-aliased fit.
    aa: Option<AaInput<'a>>,
    /// Rendered colour per pixel. Only pixels crossed by an edge are redrawn,
    /// so a pixel an edge has left keeps its last blend (kept stale on purpose).
    render: RefCell<Vec<[f32; 4]>>,
    /// Anti-aliased images: neighbour-edge index and per-vertex marks.
    inv: RefCell<Option<AntiInversion>>,
    /// Contours whose turning angle was off in the last potential evaluation.
    inverted: RefCell<BTreeSet<usize>>,
    /// Weights of the anchor term and the marked-vertex prior.
    anti_meas: f64,
    anti_prior: f64,
    optimizer_type: i32,
}

/// Inputs of the anti-aliased fit: the source image, the final pixel labels (one contour per
/// segment) and each segment's colour.
pub struct AaInput<'a> {
    pub image: &'a RgbaImage,
    pub labels: &'a [u32],
    pub colors: Vec<[u8; 4]>,
}

fn rgba_f32(c: [u8; 4]) -> [f32; 4] {
    c.map(|v| f32::from(v) * INV_255_F32)
}

/// Contours touching one pixel: (contour, area inside the pixel, its edge pieces there).
type PixelFragments = Vec<(usize, f64, Vec<Piece>)>;

/// Type 3 prior: `w·(L2/n2 − L1/n1)² + sqrt(2 − 2cosθ + 0.001)`; returns (value, ∂/∂a, ∂/∂b, ∂/∂c).
fn sqrt_prior(a: [f64; 2], b: [f64; 2], c: [f64; 2], n1: f64, n2: f64, w: f64) -> (f64, [[f64; 2]; 3]) {
    let d1 = [b[0] - a[0], b[1] - a[1]];
    let d2 = [c[0] - b[0], c[1] - b[1]];
    let l1 = (d1[1] * d1[1] + d1[0] * d1[0]).sqrt();
    let l2 = (d2[1] * d2[1] + d2[0] * d2[0]).sqrt();
    let (i1, i2) = (1.0 / l1, 1.0 / l2);
    let u1 = [d1[0] * i1, d1[1] * i1];
    let u2 = [d2[0] * i2, d2[1] * i2];
    let cos = u2[1] * u1[1] + u2[0] * u1[0];
    let s = ((2.0 - (cos + cos)) + 0.001).sqrt();
    let len = l2 / n2 - l1 / n1;
    let value = len * w * len + s;
    // Gradients follow: g1 = −∂P/∂d1, g2 = −∂P/∂d2.
    let is = 1.0 / s;
    let mut g1 = [(-(u1[0] * cos) + u2[0]) * i1 * is, (-(u1[1] * cos) + u2[1]) * i1 * is];
    let mut g2 = [(-(u2[0] * cos) + u1[0]) * i2 * -1.0 * is, is * (-(u2[1] * cos) + u1[1]) * i2 * -1.0];
    let t = len * w;
    let t = t + t;
    g1[0] += u1[0] * t * (1.0 / n1);
    g1[1] += (1.0 / n1) * u1[1] * t;
    g2[0] += (1.0 / n2) * u2[0] * t;
    g2[1] += (1.0 / n2) * u2[1] * t;
    // accumulates: a += g1, b −= g1 + g2, c += g2.
    (value, [g1, [-g1[0] - g2[0], -g1[1] - g2[1]], g2])
}

/// Type 0 prior: `|d2/n2 − d1/n1|²`.
fn quad_prior(a: [f64; 2], b: [f64; 2], c: [f64; 2], n1: f64, n2: f64) -> (f64, [[f64; 2]; 3]) {
    let (i1, i2) = (1.0 / n1, 1.0 / n2);
    let r = [i2 * (c[0] - b[0]) - i1 * (b[0] - a[0]), i2 * (c[1] - b[1]) - i1 * (b[1] - a[1])];
    let value = r[1] * r[1] + r[0] * r[0];
    let ga = [2.0 * r[0] * i1, 2.0 * r[1] * i1];
    let gc = [2.0 * r[0] * i2, 2.0 * r[1] * i2];
    (value, [ga, [-ga[0] - gc[0], -ga[1] - gc[1]], gc])
}

impl<'a> ContourSmoother<'a> {
    pub fn new(g: &'a BoundaryGraph, pixel_counts: &'a [i32], aa: Option<AaInput<'a>>) -> Self {
        let orig = g.vertices.iter().map(|v| [v.x as f32, v.y as f32]).collect();
        let corner: Vec<Vec<bool>> = g.contours.iter().map(|c| c.corner.clone()).collect();
        let turn_peak = g.contours.iter().map(|c| c.turn_peak.clone()).collect();
        let punctured = vec![false; g.vertices.len()];
        Self {
            g,
            pixel_counts,
            orig,
            corner,
            turn_peak,
            punctured,
            strength: 0.0,
            length_weight: 0.0,
            air_pressure: 0.0,
            measurement: 0,
            aa,
            render: RefCell::new(Vec::new()),
            inv: RefCell::new(None),
            inverted: RefCell::new(BTreeSet::new()),
            anti_meas: 0.0,
            anti_prior: 0.0,
            optimizer_type: 0,
        }
    }

    fn aa(&self) -> &AaInput<'a> {
        self.aa.as_ref().expect("the anti-aliased fit needs the image, labels and colours")
    }

    /// Start the rendering from each pixel's segment colour.
    fn init_render(&self) {
        let aa = self.aa();
        *self.render.borrow_mut() = aa.labels.iter().map(|&l| rgba_f32(aa.colors[l as usize])).collect();
    }

    /// Potential and (when `grad` is given) gradient of the measurement term: redraw
    /// every pixel an edge passes through as the area-weighted mix of the segment colours and
    /// return ½·Σ|render − image|² over the whole image.
    fn render(&self, x: &[f64], mut grad: Option<&mut [f64]>) -> f64 {
        let aa = self.aa();
        let (w, h) = (self.g.width as i64, self.g.height as i64);
        if grad.is_none() {
            let mut inv = self.inverted.borrow_mut();
            inv.clear();
            inv.extend((0..self.g.contours.len()).filter(|&k| inversion::is_inverted(self.g, x, k)));
        }
        // Crossed pixels are visited in raster order, .
        let mut frags: BTreeMap<usize, PixelFragments> = BTreeMap::new();
        for (k, c) in self.g.contours.iter().enumerate() {
            let pts: Vec<[f64; 2]> = c.vertices.iter().map(|&v| Self::pos(x, v)).collect();
            for ((cx, cy), cell) in polygon_cells(&pts) {
                if cx < 0 || cy < 0 || cx >= w || cy >= h {
                    continue;
                }
                frags.entry((cy * w + cx) as usize).or_default().push((k, cell.area, cell.pieces));
            }
        }
        let col = |k: usize| rgba_f32(aa.colors[k]);
        let img = |p: usize| {
            let c = aa.image.pixels[p];
            rgba_f32([c.r, c.g, c.b, c.a])
        };
        let mut render = self.render.borrow_mut();
        for (&p, list) in &frags {
            let (ok, fill) = self.redraw(&mut render[p], list);
            if !ok {
                render[p] = img(p);
                continue;
            }
            let Some(g) = grad.as_deref_mut() else { continue };
            let iv = img(p);
            let res = [0, 1, 2, 3].map(|i| render[p][i] - iv[i]);
            for (k, _, pieces) in list {
                let mut d = col(*k);
                if let (Some(_), Some(par)) = (fill, self.g.contours[*k].parent) {
                    let pc = col(par);
                    d = [0, 1, 2, 3].map(|i| d[i] - pc[i]);
                }
                let s = f64::from((res[3] * d[3] + res[2] * d[2] + res[1] * d[1] + res[0] * d[0]) * 0.5);
                let vs = &self.g.contours[*k].vertices;
                for pc in pieces {
                    let (va, vb) = (vs[pc.edge], vs[(pc.edge + 1) % vs.len()]);
                    let (ga, gb) = piece_gradient(Self::pos(x, va), Self::pos(x, vb), pc.t0, pc.t1);
                    g[2 * va] += 2.0 * s * ga[0];
                    g[2 * va + 1] += 2.0 * s * ga[1];
                    g[2 * vb] += 2.0 * s * gb[0];
                    g[2 * vb + 1] += 2.0 * s * gb[1];
                }
            }
        }
        let mut e = 0.0f32;
        for (p, r) in render.iter().enumerate() {
            let iv = img(p);
            let d = [0, 1, 2, 3].map(|i| r[i] - iv[i]);
            e += d[3] * d[3] + d[2] * d[2] + d[1] * d[1] + d[0] * d[0];
        }
        f64::from(e) * 0.5
    }

    /// Colour of one pixel from the contours crossing it (in contour order).
    /// Children's areas are also inside their parent's polygon, so when the areas do not already
    /// add up to 1 each child's area is taken off its parent; a parent with no edge in the pixel
    /// fills it completely. Returns false (the pixel falls back to the image colour and gives no
    /// gradient) when the areas stay inconsistent or a fragment is inverted, and the contour
    /// used as fill, if any.
    fn redraw(&self, out: &mut [f32; 4], list: &PixelFragments) -> (bool, Option<usize>) {
        let aa = self.aa();
        let col = |k: usize| rgba_f32(aa.colors[k]);
        *out = [0.0; 4];
        let mut sum = 0.0f64;
        let mut bad = false;
        let mut ids: Vec<usize> = Vec::with_capacity(list.len() + 1);
        let mut areas: Vec<f64> = Vec::with_capacity(list.len() + 1);
        for (k, a, _) in list {
            let af = *a as f32;
            sum += *a;
            let c = col(*k);
            for i in 0..4 {
                out[i] += af * c[i];
            }
            ids.push(*k);
            areas.push(*a);
            if *a < 0.0 && self.g.contours[*k].vertices.len() > 4 {
                bad = true;
            }
        }
        if (sum - 1.0).abs() <= 0.001 {
            return (!bad, None);
        }
        if bad {
            return (false, None);
        }
        let n = ids.len();
        let mut fill: Option<usize> = None;
        for &k in &ids[..n] {
            if let Some(par) = self.g.contours[k].parent {
                if !ids.contains(&par) && fill.is_none() {
                    fill = Some(par);
                }
            }
        }
        if let Some(f) = fill {
            sum += 1.0;
            let c = col(f);
            for i in 0..4 {
                out[i] += c[i];
            }
            ids.push(f);
            areas.push(1.0);
        }
        for j in 0..n {
            let a = areas[j];
            match self.g.contours[ids[j]].parent {
                Some(par) => {
                    if let Some(pi) = ids.iter().position(|&q| q == par) {
                        let c = col(par);
                        let af = a as f32;
                        for i in 0..4 {
                            out[i] -= c[i] * af;
                        }
                        areas[pi] -= a;
                        sum -= a;
                    }
                }
                // Top-level fragment beside a fill: the area correction is not applied,
                // so only the colour and the sum change.
                None => {
                    if let Some(f) = fill {
                        let c = col(f);
                        let af = a as f32;
                        for i in 0..4 {
                            out[i] -= af * c[i];
                        }
                        sum -= a;
                    }
                }
            }
        }
        if areas.iter().any(|&a| a < 0.0) {
            return (false, fill);
        }
        ((sum - 1.0).abs() <= 0.001, fill)
    }

    /// Anti-aliased images start from a sub-pixel estimate of every interior
    /// vertex plus a small deterministic jitter of ±`range`.
    fn perturb(&self, x: &mut [f64], range: f64) {
        let aa = self.aa();
        let (w, h) = (self.g.width as i32, self.g.height as i32);
        let label = |px: i32, py: i32| aa.labels[(py * w + px) as usize] as i32;
        let s = 1.0 / 2.0f64.sqrt();
        let (m0, m1, m2, m3) = (-s, -s, -s, s); //
        let first: usize = self.g.border_counts.iter().sum();
        let mut rng = Lcg(1);
        for v in first..self.g.vertices.len() {
            let ox = self.orig[v][0] as i32 - 1;
            let oy = self.orig[v][1] as i32 - 1;
            let ids = [label(ox, oy), label(ox + 1, oy), label(ox, oy + 1), label(ox + 1, oy + 1)];
            // Weight of the other segments around the vertex in each of its four pixels.
            let mut foreign = [0.0f64; 4];
            for row in 0..2 {
                for c in 0..2 {
                    let (px, py) = (ox + c, oy + row);
                    if px < 0 || py < 0 || px >= w || py >= h {
                        continue;
                    }
                    let mut cands = vec![label(px, py)];
                    for k in 0..8 {
                        let (nx, ny) = (px + NBR_DX[k], py + NBR_DY[k]);
                        if nx >= 0 && ny >= 0 && nx < w && ny < h {
                            let l = label(nx, ny);
                            if !cands.contains(&l) {
                                cands.push(l);
                            }
                        }
                    }
                    let cols: Vec<[f32; 4]> = cands.iter().map(|&l| rgba_f32(aa.colors[l as usize])).collect();
                    let pc = aa.image.pixels[(py * w + px) as usize];
                    let (wts, _) = mix(&cols, rgba_f32([pc.r, pc.g, pc.b, pc.a]));
                    for &id in &ids {
                        for (j, &l) in cands.iter().enumerate().skip(1) {
                            if l == id {
                                foreign[(c + 2 * row) as usize] += wts[j];
                            }
                        }
                    }
                }
            }
            let [w0, w1, w2, w3] = foreign;
            let dx = (((w2 - w1) * m2 + (w0 - w3) * m0) * 0.5).clamp(-0.333, 0.333);
            let dy = (((w0 - w3) * m1 + (w2 - w1) * m3) * 0.5).clamp(-0.333, 0.333);
            x[2 * v] += dx;
            x[2 * v + 1] += dy;
            let (r1, r2) = (rng.next(), rng.next());
            x[2 * v] += r1 * (range + range) - range;
            x[2 * v + 1] += (range + range) * r2 - range;
        }
    }

    fn pos(x: &[f64], v: usize) -> [f64; 2] {
        [x[2 * v], x[2 * v + 1]]
    }

    fn area(&self, x: &[f64], k: usize) -> f64 {
        let vs = &self.g.contours[k].vertices;
        let n = vs.len();
        let mut a = 0.0;
        for i in 0..n {
            let p0 = Self::pos(x, vs[i]);
            let p1 = Self::pos(x, vs[(i + 1) % n]);
            a += p1[1] * p0[0] - p0[1] * p1[0];
        }
        a * 0.5
    }

    /// Displacement from the original position, computed in f32.
    fn displacement(&self, x: &[f64], v: usize) -> [f64; 2] {
        let o = self.orig[v];
        [f64::from(x[2 * v] as f32 - o[0]), f64::from(x[2 * v + 1] as f32 - o[1])]
    }

    /// Walks every (prev, vertex, next) triple that contributes a prior term.
    /// Vertices marked by the anti-inversion checks add a quadratic prior scaled by
    /// `anti_prior / strength` (the sum is multiplied by the strength afterwards).
    fn for_each_prior(&self, x: &[f64], mut f: impl FnMut([usize; 3], f64, [[f64; 2]; 3])) {
        let inv = self.inv.borrow();
        let flags = inv.as_ref().map(|i| &i.flags);
        let marked_scale = self.anti_prior / self.strength;
        for (k, c) in self.g.contours.iter().enumerate() {
            let n = c.vertices.len();
            let small = self.pixel_counts[k] < SMALL_SEGMENT_PIXELS;
            for i in 0..n {
                let ids = [c.vertices[(i + n - 1) % n], c.vertices[i], c.vertices[(i + 1) % n]];
                let n1 = f64::from(c.steps[i]);
                let n2 = f64::from(c.steps[(i + 1) % n]);
                let [a, b, cc] = ids.map(|v| Self::pos(x, v));
                if flags.is_some_and(|fl| fl[ids[1]] & inversion::PRIOR != 0) {
                    let (v, g) = quad_prior(a, b, cc, n1, n2);
                    f(ids, v * marked_scale, g.map(|p| p.map(|c| c * marked_scale)));
                }
                if small {
                    let (v, g) = quad_prior(a, b, cc, n1, n2);
                    f(ids, v, g);
                } else if !self.corner[k][i] {
                    let (v, g) = sqrt_prior(a, b, cc, n1, n2, self.length_weight);
                    f(ids, v, g);
                }
            }
        }
    }

    /// Total energy of the vertex positions `x`.
    pub fn potential(&self, x: &[f64]) -> f64 {
        let mut prior = 0.0;
        self.for_each_prior(x, |_, v, _| prior += v);
        let mut e = prior * self.strength;
        let air = self.strength * self.air_pressure;
        for k in 0..self.g.contours.len() {
            if self.pixel_counts[k] < SMALL_SEGMENT_PIXELS {
                let d = self.area(x, k) - f64::from(self.pixel_counts[k]);
                e += d * d * air;
            }
        }
        match self.measurement {
            0 => {
                for v in 0..self.g.vertices.len() {
                    let d = self.displacement(x, v);
                    e += d[1] * d[1] + d[0] * d[0];
                }
            }
            2 => {
                for v in 0..self.g.vertices.len() {
                    let d = self.displacement(x, v);
                    let over = ((d[1] * d[1] + d[0] * d[0]).sqrt() - 0.7).max(0.0);
                    e += over * over;
                }
            }
            1 => e += self.render(x, None),
            m => unimplemented!("measurement type {m} not supported"),
        }
        if let Some(inv) = self.inv.borrow().as_ref() {
            for (v, &n) in inv.anchor.iter().enumerate() {
                if n != 0 {
                    let d = self.displacement(x, v);
                    e += f64::from(n) * self.anti_meas * (d[1] * d[1] + d[0] * d[0]);
                }
            }
        }
        e
    }

    /// Gradient of the energy at `x`, written to `g`.
    pub fn gradient(&self, x: &[f64], g: &mut [f64]) {
        g.iter_mut().for_each(|v| *v = 0.0);
        self.for_each_prior(x, |ids, _, gr| {
            for (j, &v) in ids.iter().enumerate() {
                g[2 * v] += gr[j][0];
                g[2 * v + 1] += gr[j][1];
            }
        });
        for v in g.iter_mut() {
            *v *= self.strength;
        }
        let air = self.strength * self.air_pressure;
        for k in 0..self.g.contours.len() {
            if self.pixel_counts[k] >= SMALL_SEGMENT_PIXELS {
                continue;
            }
            let s = (self.area(x, k) - f64::from(self.pixel_counts[k])) * air;
            let s = s + s;
            let vs = &self.g.contours[k].vertices;
            let n = vs.len();
            for i in 0..n {
                let (a, b) = (vs[i], vs[(i + 1) % n]);
                let (pa, pb) = (Self::pos(x, a), Self::pos(x, b));
                g[2 * a] += pb[1] * 0.5 * s;
                g[2 * a + 1] += pb[0] * -0.5 * s;
                g[2 * b] += pa[1] * -0.5 * s;
                g[2 * b + 1] += pa[0] * 0.5 * s;
            }
        }
        if self.measurement == 1 {
            self.render(x, Some(&mut *g));
        }
        for v in 0..self.g.vertices.len() {
            let d = self.displacement(x, v);
            match self.measurement {
                0 => {
                    g[2 * v] += d[0] + d[0];
                    g[2 * v + 1] += d[1] + d[1];
                }
                2 => {
                    let len = (d[1] * d[1] + d[0] * d[0]).sqrt();
                    if len > 0.7 {
                        let f = 1.0 - 0.7 / len;
                        let f = f + f;
                        g[2 * v] += d[0] * f;
                        g[2 * v + 1] += d[1] * f;
                    }
                }
                1 => {}
                m => unimplemented!("measurement type {m} not supported"),
            }
        }
        if let Some(inv) = self.inv.borrow().as_ref() {
            for (v, &n) in inv.anchor.iter().enumerate() {
                if n != 0 {
                    let d = self.displacement(x, v);
                    let s = f64::from(n) * self.anti_meas;
                    let s = s + s;
                    g[2 * v] += s * d[0];
                    g[2 * v + 1] += s * d[1];
                }
            }
        }
        self.zero_border_gradient(g);
    }

    /// Border vertices may only slide along their image edge.
    fn zero_border_gradient(&self, g: &mut [f64]) {
        let [left, right, top, bottom] = self.g.border_counts;
        g[1] = 0.0;
        let mut i = 0;
        for _ in 0..left {
            g[2 * i] = 0.0;
            i += 1;
        }
        g[2 * i - 1] = 0.0;
        g[2 * i + 1] = 0.0;
        for _ in 0..right {
            g[2 * i] = 0.0;
            i += 1;
        }
        g[2 * i - 1] = 0.0;
        for _ in 0..top {
            g[2 * i + 1] = 0.0;
            i += 1;
        }
        for _ in 0..bottom {
            g[2 * i + 1] = 0.0;
            i += 1;
        }
    }

    /// Snap border vertices, clamp interior vertices.
    fn after_step(&self, x: &mut [f64]) {
        let (w, h) = (self.g.width as f64, self.g.height as f64);
        let [left, right, top, bottom] = self.g.border_counts;
        let mut i = 0;
        x[1] = 0.0;
        for _ in 0..left {
            x[2 * i] = 0.0;
            i += 1;
        }
        x[2 * i - 1] = h;
        x[2 * i + 1] = 0.0;
        for _ in 0..right {
            x[2 * i] = w;
            i += 1;
        }
        x[2 * i - 1] = h;
        for _ in 0..top {
            x[2 * i + 1] = 0.0;
            i += 1;
        }
        for _ in 0..bottom {
            x[2 * i + 1] = h;
            i += 1;
        }
        let (mx, my) = (w - EDGE_MARGIN, h - EDGE_MARGIN);
        for v in i..self.g.vertices.len() {
            x[2 * v] = x[2 * v].max(EDGE_MARGIN).min(mx);
            x[2 * v + 1] = x[2 * v + 1].max(EDGE_MARGIN).min(my);
        }
    }

    fn set_phase(&mut self, cfg: &EngineConfig, p: usize) {
        self.strength = cfg.prior_strengths[p];
        self.length_weight = cfg.length_penalty_weights[p];
        self.air_pressure = cfg.air_pressure_weight;
        self.measurement = cfg.measurement_types[p];
        self.optimizer_type = cfg.optimizer_type[p];
        self.anti_meas = cfg.anti_inv_pot_meas_scale;
        self.anti_prior = cfg.anti_inv_pot_prior_scale;
    }

    /// Hook run after every line search (anti-inversion checks and perturbation).
    fn after_line_search(&self, x: &[f64], iter: i32) -> AfterSearch {
        let mut guard = self.inv.borrow_mut();
        let Some(inv) = guard.as_mut() else { return AfterSearch::Continue };
        // Re-check the contours the last potential found inverted, then forget them.
        let check_inverted = |inv: &mut AntiInversion| {
            let set = std::mem::take(&mut *self.inverted.borrow_mut());
            set.into_iter().map(|k| inv.check_contour(self.g, x, k)).sum::<i32>() > 0
        };
        if self.optimizer_type == 0 {
            let hit = if iter % CHECK_PERIOD == CHECK_PERIOD - 1 {
                inv.full_check(self.g, x)
            } else {
                i32::from(check_inverted(inv))
            };
            if hit != 0 { AfterSearch::Restart } else { AfterSearch::Continue }
        } else {
            let n = inv.decay_anchors() + i32::from(check_inverted(inv));
            if n != 0 { AfterSearch::Reevaluate } else { AfterSearch::Continue }
        }
    }

    /// Classify each vertex of contours with ≥7 vertices from a 7-vertex window.
    fn puncture_corners(&mut self, x: &[f64]) {
        for (k, c) in self.g.contours.iter().enumerate() {
            let n = c.vertices.len();
            if n < 7 {
                continue;
            }
            for j in 0..n {
                let w: [usize; 7] = std::array::from_fn(|i| c.vertices[(j + n + i - 3) % n]);
                let mut len = [0.0f64; 6];
                let mut u = [[0.0f64; 2]; 6];
                for e in 0..6 {
                    let (a, b) = (Self::pos(x, w[e]), Self::pos(x, w[e + 1]));
                    let d = [b[0] - a[0], b[1] - a[1]];
                    let l = (d[0] * d[0] + d[1] * d[1]).sqrt();
                    len[e] = l;
                    u[e] = if l > 0.0 { [d[0] * (1.0 / l), d[1] * (1.0 / l)] } else { d };
                }
                // T[k] = 2 − 2·u_k·u_{k+1}: turn at window vertex k+1 (T[2] is the centre).
                let t: [f64; 5] = std::array::from_fn(|i| 2.0 - (u[i][0] * u[i + 1][0] + u[i][1] * u[i + 1][1]) * 2.0);
                if let Some(is_corner) = corner_features(&t, &len).map(|(f, peak)| {
                    self.turn_peak[k][j] |= peak;
                    is_corner_tree(&f)
                }) {
                    if is_corner {
                        self.punctured[w[3]] = true;
                        self.corner[k][j] = true;
                    }
                }
            }
        }
    }

    /// Run the enabled phases and return the smoothed vertex positions.
    pub fn run_phases(&mut self, cfg: &EngineConfig) -> Vec<[f64; 2]> {
        let mut x: Vec<f64> = self.g.vertices.iter().flat_map(|v| [v.x, v.y]).collect();
        // setup.
        if cfg.measurement_types.contains(&1) {
            self.init_render();
        }
        if cfg.is_anti_aliased != 0 {
            *self.inv.borrow_mut() = Some(AntiInversion::new(self.g, &x));
            self.perturb(&mut x, cfg.perturbation_range);
        }
        let mut blind = 0.0;
        for p in 0..3 {
            if cfg.smoother_enabled[p] == 0 {
                continue;
            }
            if cfg.prior_types[p] != 3 {
                unimplemented!("prior type {} not supported", cfg.prior_types[p]);
            }
            self.set_phase(cfg, p);
            // The optimizer object is shared by the phases; its blind step size carries over.
            let mut opt = Cg::new(&cfg.phases[p], x.len());
            opt.blind = blind;
            opt.minimize(self, &mut x);
            blind = opt.blind;
            if cfg.do_puncture_corners[p] != 0 {
                self.puncture_corners(&x);
            }
        }
        x.chunks(2).map(|c| [c[0], c[1]]).collect()
    }
}

/// Feature vector of a corner window (21 doubles).
/// Returns the features and whether the centre turn is the window maximum (edge entry bit 3).
fn corner_features(t: &[f64; 5], len: &[f64; 6]) -> Option<([f64; 21], bool)> {
    let mut f = [0.0f64; 21];
    f[0] = ((t[0] + t[4]) + 0.0) + (t[1] + t[3]);
    f[1] = 0.0 + t[3] + t[1];
    // The side whose immediate neighbour turns less goes to F5/F4, the other to F7/F8.
    if t[1] > t[3] {
        (f[4], f[5], f[7], f[8]) = (t[4], t[3], t[1], t[0]);
    } else {
        (f[4], f[5], f[7], f[8]) = (t[0], t[1], t[3], t[4]);
    }
    f[6] = t[2];
    f[9..15].copy_from_slice(len);
    let tmax = t.iter().copied().fold(t[0], f64::max);
    let peak = t[2] == tmax;
    f[15] = if peak { 1.0 } else { 0.0 };
    let mut nb = [t[0], t[1], t[3], t[4]];
    f[16] = nb.iter().copied().fold(nb[0], f64::max);
    let (sq, s) = nb.iter().fold((0.0, 0.0), |(q, s), v| (q + v * v, s + v));
    f[18] = (sq * 0.25 - s * 0.25 * (s * 0.25)).sqrt();
    f[17] = quickselect(&mut nb, 1);
    let mut y = [f[4], f[5], 0.0];
    f[3] = y.iter().sum();
    y[2] = f[8];
    let (sq, s) = y.iter().fold((0.0, 0.0), |(q, s), v| (q + v * v, s + v));
    f[20] = (sq * (1.0 / 3.0) - s * (1.0 / 3.0) * (s * (1.0 / 3.0))).sqrt();
    f[2] = y.iter().sum();
    f[19] = quickselect(&mut y, 1);
    Some((f, peak))
}

/// Hoare-partition quickselect of the k-th smallest element (in place).
fn quickselect(a: &mut [f64], k: usize) -> f64 {
    let (mut lo, mut hi) = (0usize, a.len() - 1);
    while lo < hi {
        let pivot = a[k];
        let (mut i, mut j) = (lo as isize, hi as isize);
        loop {
            while a[i as usize] < pivot {
                i += 1;
            }
            while pivot < a[j as usize] {
                j -= 1;
            }
            if j < i {
                break;
            }
            a.swap(i as usize, j as usize);
            i += 1;
            j -= 1;
            if j < i {
                break;
            }
        }
        if (j as usize) < k || j < 0 {
            lo = i as usize;
        }
        if k < i as usize {
            hi = j.max(0) as usize;
        }
    }
    a[k]
}

/// Learned decision tree deciding whether the centre vertex is a corner.
fn is_corner_tree(f: &[f64; 21]) -> bool {
    if 0.611194 <= f[6] {
        if 0.5 <= f[15] {
            if f[16] < 0.097537 {
                return 1.42496 <= f[6] || f[2] < 0.122841 || f[0] < 0.445642 || f[8] < 0.0046405;
            }
            false
        } else {
            1.72002 <= f[6]
        }
    } else if 0.282594 <= f[6] && f[0] < 0.322933 {
        f[0] < 0.0286571 || 0.483023 <= f[6] || f[2] < 0.00209524
    } else {
        false
    }
}

/// Every this many iterations the hook re-checks all contours.
const CHECK_PERIOD: i32 = 10;

/// What the anti-inversion hook asks of the optimizer.
enum AfterSearch {
    Continue,
    /// The energy changed; re-evaluate it and restart along −g.
    Restart,
    /// Re-evaluate the energy.
    Reevaluate,
}

/// The conjugate-gradient optimizer state.
struct Cg<'s> {
    s: &'s CgSettings,
    eps: f64,
    h: Vec<f64>,
    g: Vec<f64>,
    f: f64,
    /// Step length of blind steps (line search 5).
    blind: f64,
    /// Position in the blind/seeing cycle.
    cycle: i32,
}

impl<'s> Cg<'s> {
    fn new(s: &'s CgSettings, n: usize) -> Self {
        Self { s, eps: s.max_eps, h: vec![0.0; n], g: vec![0.0; n], f: 0.0, blind: 0.0, cycle: 0 }
    }

    /// X += t·h, then the problem's after-step hook.
    fn step(&self, p: &ContourSmoother, x: &mut [f64], t: f64) {
        for (xi, hi) in x.iter_mut().zip(&self.h) {
            *xi += t * hi;
        }
        p.after_step(x);
    }

    /// Max |h|, then the line search of the configured type.
    fn line_search(&mut self, p: &ContourSmoother, x: &mut [f64], iter: i32) {
        let max_step = self.s.max_step_size;
        let mut hmax = 0.0;
        if max_step > 0.0 {
            hmax = self.h.iter().fold(0.0f64, |m, v| m.max(v.abs()));
        }
        match self.s.line_search_type {
            0 | 2 => {
                self.quadratic_search(p, x, hmax);
            }
            5 => self.blind_search(p, x, hmax, iter),
            t => unimplemented!("line search type {t} not supported"),
        }
    }

    /// Line search type 5: in every cycle of `num_seeing_steps + num_blind_steps` line
    /// searches, the first `num_seeing_steps` are quadratic fits and the rest are blind steps of
    /// 0.8 × a running average of the fitted step lengths; a blind step that raises the
    /// potential is undone and restarts the cycle.
    fn blind_search(&mut self, p: &ContourSmoother, x: &mut [f64], hmax: f64, iter: i32) {
        let max_step = self.s.max_step_size;
        let seeing = self.s.num_seeing_steps;
        let f_saved = self.f;
        self.cycle %= self.s.num_blind_steps + seeing;
        if seeing <= self.cycle {
            let backup = x.to_vec();
            let mut t = self.blind;
            if max_step > 0.0 {
                if max_step < t * hmax {
                    t = max_step / hmax;
                }
                if max_step < -(t * hmax) {
                    t = -(max_step / hmax);
                }
            }
            self.step(p, x, t * 0.8);
            self.f = p.potential(x);
            if f_saved < self.f {
                x.copy_from_slice(&backup);
                self.f = f_saved;
                self.cycle = 0;
            }
        }
        if self.cycle < seeing {
            let moved = self.quadratic_search(p, x, hmax);
            let blend = self.s.blind_blend_latest;
            let kept = (1.0 - blend) * self.blind;
            self.blind = if iter == 0 { moved } else { moved * blend + kept };
        }
        self.cycle += 1;
    }

    /// Quadratic-fit line search (type 0). Returns the total distance moved
    /// along h.
    fn quadratic_search(&mut self, p: &ContourSmoother, x: &mut [f64], hmax: f64) -> f64 {
        let max_step = self.s.max_step_size;
        if max_step > 0.0 && max_step < hmax * self.eps {
            self.eps = max_step / hmax;
        }
        let saved_eps = self.eps;
        let mut total = 0.0;
        let mut iter = 0;
        loop {
            let f0 = self.f;
            let eps = self.eps;
            self.step(p, x, eps);
            let f1 = p.potential(x);
            self.f = f1;
            self.step(p, x, eps);
            let f2 = p.potential(x);
            self.f = f2;
            let a = ((f2 - (f1 + f1)) + f0) / (eps * eps + eps * eps);
            let mut t = if a <= 0.0 {
                0.0
            } else {
                let ae = a * eps;
                (-1.0 / (a + a)) * (ae + ae + (f2 - f0) / (eps + eps))
            };
            if max_step > 0.0 {
                if max_step < t * hmax {
                    t = max_step / hmax;
                }
                if max_step < -(t * hmax) {
                    t = (-1.0 / hmax) * max_step;
                }
            }
            let fnew = if t == 0.0 {
                f0 + 1.0
            } else {
                self.step(p, x, t);
                self.f = p.potential(x);
                self.f
            };
            let (mut which, mut best) = if f1 < f0 { (1u8, f1) } else { (0u8, f0) };
            if f2 < best {
                which = 2;
                best = f2;
            }
            let moved = if fnew < best {
                eps + eps + t
            } else {
                let back = f64::from(which) * eps;
                self.step(p, x, -((eps + eps + t) - back));
                self.f = p.potential(x);
                back
            };
            total += moved;
            // The ε growth factor is not in the text config; irrelevant with
            // quad_line_srch_max_iter = 1 and use_eps_from_step = 1 (all presets).
            self.eps = self.eps.max(self.s.min_eps).min(self.s.max_eps);
            iter += 1;
            let keep_going = f0 != 0.0
                && self.s.quad_line_srch_rel_tol < (f0 - self.f) / f0
                && iter < self.s.quad_line_srch_max_iter;
            if !keep_going {
                break;
            }
        }
        self.eps = if self.s.use_eps_from_step != 0 {
            (total.abs() * self.s.step_fraction).max(self.s.min_eps).min(self.s.max_eps)
        } else {
            saved_eps
        };
        total
    }

    /// →.
    fn minimize(&mut self, p: &ContourSmoother, x: &mut [f64]) {
        self.f = p.potential(x);
        p.gradient(x, &mut self.g);
        let mut gg = 0.0;
        for (hi, gi) in self.h.iter_mut().zip(&self.g) {
            *hi = -gi;
            gg += *hi * *hi;
        }
        let mut fprev = self.f;
        let mut stall = 0;
        self.cycle = 0;
        for iter in 0..self.s.max_iter {
            fprev = fprev.min(self.f);
            self.line_search(p, x, iter);
            match p.after_line_search(x, iter) {
                AfterSearch::Continue => {}
                AfterSearch::Restart => {
                    // → restart hook →; h = −g uses
                    // the gradient from before the line search.
                    self.f = p.potential(x);
                    fprev = self.f;
                    for (hi, gi) in self.h.iter_mut().zip(&self.g) {
                        *hi = -gi;
                    }
                }
                AfterSearch::Reevaluate => {
                    self.f = p.potential(x);
                    fprev = self.f;
                }
            }
            let df = fprev - self.f;
            let rel = if fprev <= 0.0 { 0.0 } else { df / fprev };
            if rel < self.s.rel_tol || (df < self.s.abs_tol && 0.0 <= df) {
                stall += 1;
                if self.s.knock_out_count_down < stall && self.s.min_iter < iter {
                    break;
                }
            } else {
                stall = 0;
            }
            p.gradient(x, &mut self.g);
            let gg_old = gg;
            gg = self.g.iter().map(|v| v * v).sum();
            let beta = if 1e-30 <= gg_old.abs() { gg / gg_old } else { 0.0 };
            let mut descent = 0.0;
            for (hi, gi) in self.h.iter_mut().zip(&self.g) {
                *hi = beta * *hi - gi;
                descent -= gi * *hi;
            }
            if descent <= 0.0 || iter % self.s.iter_for_restart == 0 {
                // Restart along steepest descent.
                for (hi, gi) in self.h.iter_mut().zip(&self.g) {
                    *hi = -gi;
                }
            }
        }
    }
}

/// Smooth the boundary graph in place with the given engine configuration.
pub fn smooth(g: &mut BoundaryGraph, pixel_counts: &[i32], cfg: &EngineConfig, aa: Option<AaInput>) {
    let mut s = ContourSmoother::new(g, pixel_counts, aa);
    let pos = s.run_phases(cfg);
    let (corner, turn_peak, punctured) = (s.corner, s.turn_peak, s.punctured);
    for (v, p) in g.vertices.iter_mut().zip(pos) {
        v.x = p[0];
        v.y = p[1];
    }
    for (v, &pc) in g.vertices.iter_mut().zip(&punctured) {
        v.fixed |= pc;
    }
    for ((c, cf), tp) in g.contours.iter_mut().zip(corner).zip(turn_peak) {
        c.corner = cf;
        c.turn_peak = tp;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn numeric_grad(f: impl Fn([f64; 2], [f64; 2], [f64; 2]) -> f64, pts: [[f64; 2]; 3]) -> [[f64; 2]; 3] {
        let mut out = [[0.0; 2]; 3];
        let h = 1e-6;
        for i in 0..3 {
            for j in 0..2 {
                let mut p = pts;
                p[i][j] += h;
                let up = f(p[0], p[1], p[2]);
                p[i][j] -= 2.0 * h;
                let dn = f(p[0], p[1], p[2]);
                out[i][j] = (up - dn) / (2.0 * h);
            }
        }
        out
    }

    #[test]
    fn prior_gradients_match_numeric() {
        let pts = [[0.0, 0.0], [1.0, 0.3], [2.2, 1.1]];
        let (_, g) = sqrt_prior(pts[0], pts[1], pts[2], 1.0, 2.0, 0.5);
        let ng = numeric_grad(|a, b, c| sqrt_prior(a, b, c, 1.0, 2.0, 0.5).0, pts);
        let (_, gq) = quad_prior(pts[0], pts[1], pts[2], 1.0, 2.0);
        let nq = numeric_grad(|a, b, c| quad_prior(a, b, c, 1.0, 2.0).0, pts);
        for i in 0..3 {
            for j in 0..2 {
                assert!((g[i][j] - ng[i][j]).abs() < 1e-5, "sqrt {i},{j}: {} vs {}", g[i][j], ng[i][j]);
                assert!((gq[i][j] - nq[i][j]).abs() < 1e-5, "quad {i},{j}");
            }
        }
    }

    #[test]
    fn quickselect_matches_sorting() {
        let mut a = [0.4, 0.1, 0.9, 0.3];
        assert_eq!(quickselect(&mut a, 1), 0.3);
        let mut b = [0.5, 0.5, 0.2];
        assert_eq!(quickselect(&mut b, 1), 0.5);
    }

    #[test]
    fn square_corners_are_punctured() {
        use crate::params::{ImageType, Quality};
        let (w, h) = (40usize, 40usize);
        let mut l = vec![0u32; w * h];
        for y in 8..32 {
            for x in 8..32 {
                l[y * w + x] = 1;
            }
        }
        let mut g = super::super::boundary::trace_boundaries(w, h, &l, 2, true);
        let cfg = EngineConfig::preset(ImageType::ArtworkUnblended, Quality::Medium);
        smooth(&mut g, &[w as i32 * h as i32 - 576, 576], &cfg, None);
        let c = &g.contours[1];
        let corners: Vec<(f64, f64)> = (0..c.vertices.len())
            .filter(|&i| c.corner[i])
            .map(|i| (g.vertices[c.vertices[i]].x.round(), g.vertices[c.vertices[i]].y.round()))
            .collect();
        assert_eq!(corners.len(), 4, "{corners:?}");
        for p in [(8.0, 8.0), (32.0, 8.0), (32.0, 32.0), (8.0, 32.0)] {
            assert!(corners.contains(&p), "missing {p:?} in {corners:?}");
        }
    }

    #[test]
    fn contour_area_is_pixel_count() {
        // Confirms the tracer's orientation matches the air-pressure sign convention.
        let mut l = vec![0u32; 10 * 10];
        for y in 2..6 {
            for x in 3..8 {
                l[y * 10 + x] = 1;
            }
        }
        let g = super::super::boundary::trace_boundaries(10, 10, &l, 2, false);
        let counts = [80, 20];
        let s = ContourSmoother::new(&g, &counts, None);
        let x: Vec<f64> = g.vertices.iter().flat_map(|v| [v.x, v.y]).collect();
        assert_eq!(s.area(&x, 1), 20.0);
    }

    #[test]
    fn smoothing_rounds_a_staircase_circle() {
        use crate::params::{ImageType, Quality};
        let (w, h) = (40usize, 40usize);
        let mut l = vec![0u32; w * h];
        for y in 0..h {
            for x in 0..w {
                let (dx, dy) = (x as f64 + 0.5 - 20.0, y as f64 + 0.5 - 20.0);
                if dx * dx + dy * dy < 14.0 * 14.0 {
                    l[y * w + x] = 1;
                }
            }
        }
        let count = l.iter().filter(|&&v| v == 1).count() as i32;
        let mut g = super::super::boundary::trace_boundaries(w, h, &l, 2, true);
        let cfg = EngineConfig::preset(ImageType::ArtworkUnblended, Quality::Medium);
        smooth(&mut g, &[w as i32 * h as i32 - count, count], &cfg, None);
        // After smoothing every vertex of the circle lies close to radius 14.
        for &v in &g.contours[1].vertices {
            let (dx, dy) = (g.vertices[v].x - 20.0, g.vertices[v].y - 20.0);
            let r = (dx * dx + dy * dy).sqrt();
            assert!((r - 14.0).abs() < 0.6, "radius {r}");
        }
    }
}
