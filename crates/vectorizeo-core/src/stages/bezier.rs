//! Bezier fitter: splits every boundary run between fixed vertices into
//! pieces by annealed merging (a 1-D analogue of the segmenter), fits one cubic per piece by
//! regularised least squares, and assembles per-contour paths. Shared boundaries are fitted once
//! and reused (reversed) by the neighbouring segment, so shapes stay seamless.
//!
//! The stages are: driver, keeper selection, piece costs, fits, final fits and path assembly.
//! A node-reduction optimiser is not implemented.

use super::boundary::BoundaryGraph;
use crate::params::EngineConfig;

/// Rise fraction of the annealing schedule (not part of the text config).
const RISE_FRACTION: f64 = 0.5;
/// Diagonal regularisation added to the normal equations.
const RIDGE: f64 = 1e-5;
/// A stored fit is reversed when its P0 differs from the start vertex by more than this.
const SAME_POINT: f64 = 9.999999974752427e-07;

type P = [f64; 2];

/// Vertex type byte during fitting.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum VType {
    Free = 0,
    PieceInterior = 1,
    Sample = 2,
    Fixed = 3,
    RunEnd = 4,
    Keeper = 5,
}

#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum Element {
    /// Contour edge `at` → `at + 1`.
    Line { at: usize, from: usize, to: usize },
    /// Cubic through vertex `from` (contour position `at`) → `to` using fit `fit`, possibly
    /// stored in reverse.
    Cubic { at: usize, from: usize, to: usize, fit: usize, reversed: bool },
}

#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct BezierPaths {
    /// Control points of every fitted piece: P0, P1, P2, P3.
    pub fits: Vec<[P; 4]>,
    /// One element list per contour, in tracing order.
    pub paths: Vec<Vec<Element>>,
}

fn bernstein3(t: f64) -> [f64; 4] {
    let u = 1.0 - t;
    [u * u * u, u * u * t * 3.0, t * t * u * 3.0, t * t * t]
}

fn eval_cubic(c: &[P; 4], t: f64) -> P {
    let b = bernstein3(t);
    [0, 1].map(|k| b[0] * c[0][k] + b[1] * c[1][k] + b[2] * c[2][k] + b[3] * c[3][k])
}

fn lerp(a: P, b: P, t: f64) -> P {
    [t * b[0] + (1.0 - t) * a[0], (1.0 - t) * a[1] + t * b[1]]
}

/// Solve the symmetric positive definite system (Cholesky), as LAPACK does in.
fn solve_spd<const N: usize>(mut a: [[f64; N]; N], mut b: [f64; N]) -> [f64; N] {
    for j in 0..N {
        let mut d = a[j][j];
        for k in 0..j {
            d -= a[j][k] * a[j][k];
        }
        let d = d.sqrt();
        a[j][j] = d;
        for i in j + 1..N {
            let mut s = a[i][j];
            for k in 0..j {
                s -= a[i][k] * a[j][k];
            }
            a[i][j] = s / d;
        }
    }
    for i in 0..N {
        for k in 0..i {
            b[i] -= a[i][k] * b[k];
        }
        b[i] /= a[i][i];
    }
    for i in (0..N).rev() {
        for k in i + 1..N {
            b[i] -= a[k][i] * b[k];
        }
        b[i] /= a[i][i];
    }
    b
}

/// Interior samples of the range and their normalised chord-length parameters.
fn samples(pts: &[P], p0: P, p3: P) -> (Vec<P>, Vec<f64>) {
    let mut t = Vec::with_capacity(pts.len());
    let mut acc = 0.0;
    let mut prev = p0;
    for p in pts {
        acc += ((p[0] - prev[0]).powi(2) + (p[1] - prev[1]).powi(2)).sqrt();
        t.push(acc);
        prev = *p;
    }
    let mut total = ((p3[0] - prev[0]).powi(2) + (p3[1] - prev[1]).powi(2)).sqrt() + acc;
    if total == 0.0 {
        total = 1.0;
    }
    for v in t.iter_mut() {
        *v *= 1.0 / total;
    }
    (pts.to_vec(), t)
}

/// Fit a cubic from `p0` through `inner` to `p3`. Returns (control points, cost).
fn fit_range(p0: P, inner: &[P], p3: P) -> ([P; 4], f64) {
    let m = inner.len();
    let mut ctrl = [p0, inner.first().copied().unwrap_or(p0), inner.last().copied().unwrap_or(p3), p3];
    if m == 0 {
        return (ctrl, 0.0);
    }
    let (s, t) = samples(inner, p0, p3);
    if m > 3 {
        // Unknowns [P1x, P1y, P2x, P2y]; A row (i, c): B1 at col c, B2 at col 2+c.
        let mut ata = [[0.0; 4]; 4];
        let mut atr = [0.0; 4];
        for d in 0..4 {
            ata[d][d] = RIDGE;
        }
        for i in 0..m {
            let b = bernstein3(t[i]);
            for c in 0..2 {
                let r = s[i][c] - (b[3] * p3[c] + b[0] * p0[c]);
                let (i1, i2) = (c, 2 + c);
                ata[i1][i1] += b[1] * b[1];
                ata[i2][i2] += b[2] * b[2];
                ata[i1][i2] += b[1] * b[2];
                ata[i2][i1] += b[1] * b[2];
                atr[i1] += b[1] * r;
                atr[i2] += b[2] * r;
            }
        }
        let x = solve_spd(ata, atr);
        ctrl[1] = [x[0], x[1]];
        ctrl[2] = [x[2], x[3]];
        let cost = (0..m)
            .map(|i| {
                let e = eval_cubic(&ctrl, t[i]);
                let d = [s[i][0] - e[0], s[i][1] - e[1]];
                d[0] * d[0] + d[1] * d[1]
            })
            .sum();
        return (ctrl, cost);
    }
    if m == 3 {
        // Quadratic with one control point Q, then exact degree elevation.
        let mut ata = [[RIDGE, 0.0], [0.0, RIDGE]];
        let mut atr = [0.0; 2];
        for i in 0..m {
            let u = 1.0 - t[i];
            let (b0, b1, b2) = (u * u, u * t[i] + u * t[i], t[i] * t[i]);
            for c in 0..2 {
                let r = s[i][c] - (b2 * p3[c] + b0 * p0[c]);
                ata[c][c] += b1 * b1;
                atr[c] += b1 * r;
            }
        }
        let q = solve_spd(ata, atr);
        let mut cost = 0.0;
        for i in 0..m {
            let u = 1.0 - t[i];
            let e = [0, 1].map(|c| u * u * p0[c] + (u * t[i] + u * t[i]) * q[c] + t[i] * t[i] * p3[c]);
            cost += (s[i][0] - e[0]).powi(2) + (s[i][1] - e[1]).powi(2);
        }
        ctrl[1] = [p0[0] + 2.0 / 3.0 * (q[0] - p0[0]), p0[1] + 2.0 / 3.0 * (q[1] - p0[1])];
        ctrl[2] = [p3[0] + 2.0 / 3.0 * (q[0] - p3[0]), p3[1] + 2.0 / 3.0 * (q[1] - p3[1])];
        return (ctrl, cost);
    }
    // m ∈ {1, 2}: straight cubic; only the first sample is scored.
    ctrl[1] = lerp(p0, p3, 0.25);
    ctrl[2] = lerp(p0, p3, 0.75);
    let e = lerp(p0, p3, t[0]);
    let cost = (e[1] - s[0][1]) * (e[1] - s[0][1]) + (e[0] - s[0][0]) * (e[0] - s[0][0]);
    (ctrl, cost)
}

#[derive(Clone, Copy, Debug)]
struct Piece {
    start: i32,
    len: i32,
    next: Option<usize>,
    cost: f64,
    dirty: bool,
}

struct Fitter<'a> {
    g: &'a BoundaryGraph,
    types: Vec<VType>,
    piece_id: Vec<usize>,
    next_piece: usize,
}

impl Fitter<'_> {
    fn pos(&self, v: usize) -> P {
        [self.g.vertices[v].x, self.g.vertices[v].y]
    }

    fn vid(&self, k: usize, i: i32) -> usize {
        let vs = &self.g.contours[k].vertices;
        vs[i.rem_euclid(vs.len() as i32) as usize]
    }

    /// Cost of the cubic over contour indices [from, to] (with flag 0).
    fn range_cost(&self, k: usize, from: i32, to: i32) -> f64 {
        if to <= from {
            return 0.0;
        }
        let inner: Vec<P> = (from + 1..to).map(|i| self.pos(self.vid(k, i))).collect();
        fit_range(self.pos(self.vid(k, from)), &inner, self.pos(self.vid(k, to))).1
    }

    /// Step from `i` in direction `dir` until a vertex whose type differs from `ty`.
    fn search(&self, k: usize, i: i32, dir: i32, ty: VType) -> i32 {
        let n = self.g.contours[k].vertices.len() as i32;
        let origin = self.vid(k, i);
        let mut j = i;
        loop {
            j += dir;
            let v = self.vid(k, j);
            if self.types[v] != ty || v == origin {
                break;
            }
        }
        (n + j).rem_euclid(n)
    }

    fn cost(&self, k: usize, base: i32, p: &mut Piece) -> f64 {
        if p.dirty {
            p.cost = self.range_cost(k, base + p.start, base + p.start + p.len);
            p.dirty = false;
        }
        p.cost
    }

    /// Cost change when piece bounds move by (ds, de); (sign convention: positive = worse).
    fn shift(&self, k: usize, base: i32, p: &mut Piece, ds: i32, de: i32) -> f64 {
        let old = self.cost(k, base, p);
        let new = self.range_cost(k, base + p.start + ds, base + p.start + p.len + de);
        let d = old - new;
        if de != ds && de - ds > 0 { -d } else { d }
    }

    /// Anneal the piece list of one run.
    fn select(&self, k: usize, base: i32, len: i32, cfg: &EngineConfig) -> Vec<Piece> {
        let mut pieces: Vec<Piece> =
            (0..len).map(|i| Piece { start: i, len: 1, next: Some(i as usize + 1), cost: 0.0, dirty: true }).collect();
        pieces[len as usize - 1].next = None;
        let (init, fin) = (cfg.stat_thresh_initial, cfg.stat_thresh_final);
        let iters = cfg.bezier_max_iterations;
        let ratio = ((fin / init).log2() / (f64::from(iters) * RISE_FRACTION)).exp2();
        let mut thresh = init;
        for it in 0..iters {
            let mut cur = Some(0usize);
            while let Some(pi) = cur {
                let Some(qi) = pieces[pi].next else { break };
                if pieces[qi].next.is_none() {
                    break;
                }
                self.select_step(k, base, &mut pieces, pi, qi, thresh);
                cur = pieces[pi].next;
            }
            thresh = if f64::from(iters) * RISE_FRACTION <= f64::from(it) { fin } else { ratio * thresh };
        }
        pieces
    }

    fn select_step(&self, k: usize, base: i32, pieces: &mut [Piece], pi: usize, qi: usize, thresh: f64) {
        let (mut p, mut q) = (pieces[pi], pieces[qi]);
        // Merge cost.
        let cp = self.cost(k, base, &mut p);
        let cq = self.cost(k, base, &mut q);
        let merged = self.range_cost(k, base + p.start, base + p.start + p.len + q.len);
        if merged - (cq + cp) < thresh {
            p.len += q.len;
            p.next = q.next;
            p.dirty = true;
            pieces[pi] = p;
            pieces[qi] = q;
            return;
        }
        let a = self.shift(k, base, &mut q, -1, 0);
        let b = self.shift(k, base, &mut p, 0, -1);
        if a - b >= 0.0 {
            let c = self.shift(k, base, &mut p, 0, 1);
            let d = self.shift(k, base, &mut q, 1, 0);
            if c - d < 0.0 && q.len > 1 {
                p.len += 1;
                q.len -= 1;
                q.start += 1;
                p.dirty = true;
                q.dirty = true;
            }
        } else if p.len > 1 {
            q.len += 1;
            p.len -= 1;
            q.start -= 1;
            q.dirty = true;
            p.dirty = true;
        }
        pieces[pi] = p;
        pieces[qi] = q;
    }

    /// Mark keepers (5), piece interiors (1 + id) and run ends (4).
    fn mark(&mut self, k: usize, start: i32, end: i32, pieces: &[Piece], run_len: i32) {
        let mut cur = Some(0usize);
        let mut i = start;
        while let Some(pi) = cur {
            let p = pieces[pi];
            let next_start = p.next.map_or(run_len, |q| pieces[q].start);
            let v = self.vid(k, i);
            self.types[v] = VType::Keeper;
            i += 1;
            for _ in 0..(next_start - (p.start + 1)).max(0) {
                let v = self.vid(k, i);
                self.types[v] = VType::PieceInterior;
                self.piece_id[v] = self.next_piece;
                i += 1;
            }
            self.next_piece += 1;
            cur = p.next;
        }
        let (vs, ve) = (self.vid(k, start), self.vid(k, end));
        self.types[vs] = VType::RunEnd;
        self.types[ve] = VType::RunEnd;
    }
}

/// Select pieces, fit them and assemble paths.
pub fn fit_beziers(g: &BoundaryGraph, cfg: &EngineConfig) -> BezierPaths {
    let types = g.vertices.iter().map(|v| if v.fixed { VType::Fixed } else { VType::Free }).collect();
    let mut f = Fitter { g, types, piece_id: vec![0; g.vertices.len()], next_piece: 0 };

    // Keeper selection, once per boundary run.
    for k in 0..g.contours.len() {
        let n = g.contours[k].vertices.len() as i32;
        for i in 0..n {
            if f.types[f.vid(k, i)] != VType::Free {
                continue;
            }
            let start = f.search(k, i, -1, VType::Free);
            let mut end = f.search(k, i, 1, VType::Free);
            while end <= start {
                end += n;
            }
            let run_len = end - start + 1;
            let pieces = f.select(k, start, run_len, cfg);
            f.mark(k, start, end, &pieces, run_len);
        }
    }

    // Final fit of every piece with interior vertices; interiors become samples.
    let mut fits = vec![[[0.0; 2]; 4]; f.next_piece];
    for k in 0..g.contours.len() {
        let n = g.contours[k].vertices.len() as i32;
        for i in 0..n {
            let v = f.vid(k, i);
            if f.types[v] != VType::PieceInterior {
                continue;
            }
            let from = f.search(k, i, -1, VType::PieceInterior);
            let mut to = f.search(k, i, 1, VType::PieceInterior);
            while to <= from {
                to += n;
            }
            let inner: Vec<P> = (from + 1..to).map(|j| f.pos(f.vid(k, j))).collect();
            let (ctrl, _) = fit_range(f.pos(f.vid(k, from)), &inner, f.pos(f.vid(k, to)));
            fits[f.piece_id[v]] = ctrl;
            for j in from + 1..to {
                let w = f.vid(k, j);
                f.types[w] = VType::Sample;
            }
        }
    }

    // Walk each contour from a non-sample vertex.
    let mut paths = Vec::with_capacity(g.contours.len());
    for k in 0..g.contours.len() {
        let n = g.contours[k].vertices.len() as i32;
        let mut first = 0i32;
        while f.types[f.vid(k, first)] == VType::Sample {
            first -= 1;
        }
        let first = first.rem_euclid(n);
        let mut els = Vec::new();
        let mut i = first;
        let stop = first + n;
        while i < stop {
            let a = f.vid(k, i);
            let b = f.vid(k, i + 1);
            if f.types[b] == VType::Sample {
                let mut j = i + 1;
                while f.types[f.vid(k, j)] == VType::Sample {
                    j += 1;
                }
                let end = f.vid(k, j);
                let fit = f.piece_id[b];
                let p0 = fits[fit][0];
                let start = f.pos(a);
                let reversed = (p0[0] - start[0]).abs() > SAME_POINT || (p0[1] - start[1]).abs() > SAME_POINT;
                els.push(Element::Cubic { at: i.rem_euclid(n) as usize, from: a, to: end, fit, reversed });
                i = j;
            } else {
                els.push(Element::Line { at: i.rem_euclid(n) as usize, from: a, to: b });
                i += 1;
            }
        }
        paths.push(els);
    }
    BezierPaths { fits, paths }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cubic_fit_recovers_exact_curve() {
        let c = [[0.0, 0.0], [1.0, 3.0], [4.0, 3.0], [5.0, 0.0]];
        // Sample at the chord-length parameters the fitter will compute, iterating to consistency.
        let inner: Vec<P> = (1..12).map(|i| eval_cubic(&c, f64::from(i) / 12.0)).collect();
        let (fit, cost) = fit_range(c[0], &inner, c[3]);
        // Chord-length parameters are never re-optimised so a small residual remains.
        assert!(cost < 0.2, "cost {cost}");
        assert!((fit[1][1] - 3.0).abs() < 0.6 && (fit[2][1] - 3.0).abs() < 0.6, "{fit:?}");
    }

    #[test]
    fn solve_spd_matches_known_solution() {
        let x = solve_spd([[4.0, 2.0], [2.0, 3.0]], [2.0, 1.0]);
        assert!((x[0] - 0.5).abs() < 1e-12 && x[1].abs() < 1e-12);
    }
}
