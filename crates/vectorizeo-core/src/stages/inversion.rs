//! Anti-inversion checks of the anti-aliased contour smoother (hook).
//!
//! Each vertex gets a list of nearby edges once, before the perturbation. After
//! line searches the smoother re-checks contours: an edge crossing another edge,
//! or a stretch of contour whose winding no longer adds up, marks its vertices. A marked vertex
//! switches on two energy terms in the smoother: flag 0x10 (permanent) adds a quadratic prior,
//! and the anchor counter (set to 3, decremented to 1 by the periodic full check) pulls
//! it back towards its starting position.

use super::ann::KdTree;
use super::boundary::BoundaryGraph;

/// Vertex flag bits: 2 = marked in the current check, 4 = checked, 0x10 = quadratic prior on.
const MARKED: u8 = 0x02;
const CHECKED: u8 = 0x04;
pub(crate) const PRIOR: u8 = 0x10;

/// Nearest edges kept per vertex.
const MAX_NEIGHBOURS: usize = 50;
/// Squared radius of the neighbourhood.
const RADIUS2: f64 = 25.0;
/// `annkSearch` error bound passed by the engine.
const ANN_EPS: f64 = 1e-12;

type P = [f64; 2];

fn normalized(d: P) -> P {
    let len = (d[1] * d[1] + d[0] * d[0]).sqrt();
    if len > 0.0 {
        let inv = 1.0 / len;
        [d[0] * inv, inv * d[1]]
    } else {
        d
    }
}

/// `p` lies on the line through `a` along `dir` and, unless it coincides with
/// `a` or `b`, within the segment a→b.
fn touches(p: P, a: P, b: P, dir: P) -> bool {
    let dx = p[0] - a[0];
    let dy = p[1] - a[1];
    if (dy * dir[0] - dx * dir[1]).abs() >= 0.0001 {
        return false;
    }
    let proj = dy * dir[1] + dx * dir[0];
    let ab = ((b[1] - a[1]) * (b[1] - a[1]) + (b[0] - a[0]) * (b[0] - a[0])).sqrt();
    let pb = (p[1] - b[1]) * (p[1] - b[1]) + (p[0] - b[0]) * (p[0] - b[0]);
    !((proj < 0.0 || ab < proj) && 1e-8 <= dy * dy + dx * dx && 1e-8 <= pb)
}

/// Signed crossing of segments p1→p2 and p3→p4: ±2 for a proper crossing, ±1 when
/// an endpoint touches the other segment, 0 otherwise (also for parallel segments).
fn segment_crossing(p1: P, p2: P, p3: P, p4: P) -> i32 {
    let mut c = (p4[1] - p3[1]) * (p2[0] - p1[0]) - (p4[0] - p3[0]) * (p2[1] - p1[1]);
    if c == 0.0 {
        return 0;
    }
    let s = if 0.0 <= c { 1 } else { -1 };
    let u = normalized([p2[0] - p1[0], p2[1] - p1[1]]);
    let v = normalized([p4[0] - p3[0], p4[1] - p3[1]]);
    if touches(p1, p3, p4, v) || touches(p2, p3, p4, v) || touches(p3, p1, p2, u) || touches(p4, p1, p2, u) {
        return s;
    }
    let mut ta = (p4[0] - p3[0]) * (p1[1] - p3[1]) - (p4[1] - p3[1]) * (p1[0] - p3[0]);
    let mut tb = (p2[0] - p1[0]) * (p1[1] - p3[1]) - (p2[1] - p1[1]) * (p1[0] - p3[0]);
    if c < 0.0 {
        c = -c;
        ta = -ta;
        tb = -tb;
    }
    if 0.0 < ta && ta < c && 0.0 < tb && tb < c { s * 2 } else { 0 }
}

/// Two corners meeting at the same vertex `c`, compared through the unit
/// directions from `c` (a→a2 against p→p2).
fn corner_crossing(a: P, a2: P, p: P, p2: P, c: P) -> i32 {
    let rel = |q: P| normalized([q[0] - c[0], q[1] - c[1]]);
    segment_crossing(rel(a), rel(a2), rel(p), rel(p2))
}

/// Signed crossing of the ray o + t·d with the segment p3→p4 (±2, ±1 when an
/// endpoint lies on the ray's line).
fn ray_crossing(o: P, d: P, p3: P, p4: P) -> i32 {
    let den = (p4[1] - p3[1]) * d[0] - (p4[0] - p3[0]) * d[1];
    if den == 0.0 {
        return 0;
    }
    if ((p4[1] - o[1]) * d[0] - (p4[0] - o[0]) * d[1]).abs() < 0.0001
        || ((p3[1] - o[1]) * d[0] - (p3[0] - o[0]) * d[1]).abs() < 0.0001
    {
        return if den <= 0.0 { -1 } else { 1 };
    }
    let t3 = (p4[0] - p3[0]) * (o[1] - p3[1]) - (p4[1] - p3[1]) * (o[0] - p3[0]);
    let t2 = (o[1] - p3[1]) * d[0] - (o[0] - p3[0]) * d[1];
    if den <= 0.0 {
        if t3 <= 0.0 && t2 < 0.0 && den < t2 {
            return -2;
        }
    } else if 0.0 <= t3 && 0.0 < t2 && t2 < den {
        return 2;
    }
    0
}

/// Unit direction pointing away from the contour at vertex `prev → p → next`
/// (the angle bisector, or the normal of the chord when the corner is nearly straight).
fn outward(prev: P, p: P, next: P) -> P {
    let u = normalized([prev[0] - p[0], prev[1] - p[1]]);
    let w = normalized([next[0] - p[0], next[1] - p[1]]);
    let (mut sx, sy) = (w[0] + u[0], w[1] + u[1]);
    let mut dx = u[1] - w[1];
    let mut dy = w[0] - u[0];
    if dy * dy + dx * dx < sy * sy + sx * sx {
        dy = sy;
        if 0.0 < w[1] * u[0] - u[1] * w[0] {
            sx = -sx;
            dy = -sy;
        }
        dx = sx;
    }
    normalized([dx, dy])
}

pub(crate) struct AntiInversion {
    /// Per vertex: nearby edges as (contour, index of the edge's start vertex).
    neighbours: Vec<Vec<(usize, usize)>>,
    /// Flag bits (see the constants).
    pub flags: Vec<u8>,
    /// anchor counter.
    pub anchor: Vec<u8>,
    /// Per contour: winding term `2 − Σ ray crossings`.
    winding: Vec<i32>,
}

impl AntiInversion {
    /// Kd-tree over every contour slot, then for each vertex
    /// the edges starting at its ≤50 nearest slots within distance 5, its own edges excluded.
    pub(crate) fn new(g: &BoundaryGraph, x: &[f64]) -> Self {
        let pos = |v: usize| [x[2 * v], x[2 * v + 1]];
        let mut slots: Vec<(usize, usize)> = Vec::new();
        let mut pts: Vec<P> = Vec::new();
        for (k, c) in g.contours.iter().enumerate() {
            for (i, &v) in c.vertices.iter().enumerate() {
                slots.push((k, i));
                pts.push(pos(v));
            }
        }
        let tree = KdTree::new(&pts);
        let nv = g.vertices.len();
        let k = nv.min(MAX_NEIGHBOURS);
        let mut neighbours = vec![Vec::new(); nv];
        let mut done = vec![false; nv];
        for c in &g.contours {
            for &v in &c.vertices {
                if done[v] {
                    continue;
                }
                done[v] = true;
                for (d2, s) in tree.search(pos(v), k, ANN_EPS) {
                    let (ck, ci) = slots[s];
                    if d2 < RADIUS2 && g.contours[ck].vertices[ci] != v {
                        neighbours[v].push((ck, ci));
                    }
                }
            }
        }
        Self { neighbours, flags: vec![0; nv], anchor: vec![0; nv], winding: vec![0; g.contours.len()] }
    }

    /// Cast a ray outwards from vertex `idx` and sum its signed crossings with
    /// the contour's other edges; returns `2 − sum` (an odd sum is logged and taken as 0).
    fn winding_at(g: &BoundaryGraph, x: &[f64], k: usize, idx: usize) -> i32 {
        let vs = &g.contours[k].vertices;
        let n = vs.len();
        let pos = |i: usize| [x[2 * vs[i % n]], x[2 * vs[i % n] + 1]];
        let o = pos(idx);
        let d = outward(pos(idx + n - 1), o, pos(idx + 1));
        let mut sum = 0;
        if n > 2 {
            for i in idx + 2..idx + n {
                sum += ray_crossing(o, d, pos(i - 1), pos(i));
            }
        }
        if sum % 2 != 0 {
            // "bad_intersection_sign_sum_parity" goes to the warning set.
            sum = 0;
        }
        2 - sum
    }

    /// Re-check contour `k`; returns the number of its vertices marked.
    pub(crate) fn check_contour(&mut self, g: &BoundaryGraph, x: &[f64], k: usize) -> i32 {
        let pos = |v: usize| [x[2 * v], x[2 * v + 1]];
        let vs = &g.contours[k].vertices;
        let n = vs.len();
        self.winding[k] = Self::winding_at(g, x, k, 0);
        for &v in vs {
            self.flags[v] &= 0xf0;
        }
        let mut marked = 0;
        let mut self_sum = 0i32;
        let mut run = 0i32;
        for i in 0..n {
            let (a, b) = (vs[i], vs[(i + 1) % n]);
            let mut crossings = 0;
            let mut own = 0;
            for &(k2, j) in &self.neighbours[a] {
                let vs2 = &g.contours[k2].vertices;
                let n2 = vs2.len();
                let (p, q) = (vs2[j], vs2[(j + 1) % n2]);
                if b == q && k == k2 {
                    // The contour passes through b twice.
                    let (a2, q2) = (vs[(i + 2) % n], vs2[(j + 2) % n2]);
                    let r = corner_crossing(pos(a), pos(a2), pos(p), pos(q2), pos(b));
                    crossings += r.abs();
                    own += r;
                } else if a != p && a != q && b != p && b != q {
                    let r = segment_crossing(pos(a), pos(b), pos(p), pos(q));
                    crossings += r.abs();
                    if k2 == k {
                        own += r;
                    }
                }
            }
            self_sum += own;
            if crossings == 0 {
                if self.winding[k] + self_sum == 0 {
                    self.flags[b] |= CHECKED;
                    run = 0;
                } else {
                    marked += i32::from(self.flags[b] & MARKED == 0);
                    self.flags[b] |= MARKED | CHECKED | PRIOR;
                    run = run.wrapping_add(1);
                    self.anchor[b] = 3;
                    if run > 20 {
                        self.winding[k] = Self::winding_at(g, x, k, i);
                        self_sum = 0;
                        run = -0x7fff_ffff;
                    }
                }
            } else {
                let old = self.flags[a];
                self.flags[a] |= MARKED | CHECKED | PRIOR;
                self.anchor[a] = 3;
                marked += i32::from(old & MARKED == 0) + i32::from(self.flags[b] & MARKED == 0);
                self.flags[b] |= MARKED | CHECKED | PRIOR;
                self.anchor[b] = 3;
            }
        }
        marked
    }

    /// Count down every anchor above 1; returns how many changed.
    pub(crate) fn decay_anchors(&mut self) -> i32 {
        let mut n = 0;
        for a in &mut self.anchor {
            if *a > 1 {
                *a -= 1;
                n += 1;
            }
        }
        n
    }

    /// Anchors count down, then every contour is re-checked.
    pub(crate) fn full_check(&mut self, g: &BoundaryGraph, x: &[f64]) -> i32 {
        let mut n = self.decay_anchors();
        for k in 0..g.contours.len() {
            n += self.check_contour(g, x, k);
        }
        n
    }
}

/// Turning-angle test: a contour whose edge directions do not turn by exactly
/// +2π (±0.001) is inverted.
pub(crate) fn is_inverted(g: &BoundaryGraph, x: &[f64], k: usize) -> bool {
    use std::f64::consts::{PI, TAU};
    let vs = &g.contours[k].vertices;
    let n = vs.len();
    let angle = |i: usize| {
        let (a, b) = (vs[i % n], vs[(i + 1) % n]);
        (x[2 * b + 1] - x[2 * a + 1]).atan2(x[2 * b] - x[2 * a])
    };
    let mut prev = angle(0);
    let mut total = 0.0;
    for i in 1..=n {
        let a = angle(i);
        let mut t = a - prev;
        if t > PI {
            t -= TAU;
        } else if t < -PI {
            t += TAU;
        }
        total += t;
        prev = a;
    }
    (total - TAU).abs() > 0.001
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn crossing_signs() {
        assert_eq!(segment_crossing([0.0, 0.0], [2.0, 2.0], [0.0, 2.0], [2.0, 0.0]).abs(), 2);
        assert_eq!(segment_crossing([0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]), 0);
        // T junction: endpoint on the other segment.
        assert_eq!(segment_crossing([0.0, 0.0], [2.0, 0.0], [1.0, 0.0], [1.0, 2.0]).abs(), 1);
        assert_eq!(ray_crossing([0.0, 0.5], [1.0, 0.0], [2.0, 0.0], [2.0, 1.0]).abs(), 2);
    }

    #[test]
    fn traced_contours_are_not_inverted() {
        let (w, h) = (12usize, 10usize);
        let mut l = vec![0u32; w * h];
        for y in 2..7 {
            for x in 3..9 {
                l[y * w + x] = if (4..6).contains(&y) && (5..7).contains(&x) { 2 } else { 1 };
            }
        }
        let g = super::super::boundary::trace_boundaries(w, h, &l, 3, true);
        let x: Vec<f64> = g.vertices.iter().flat_map(|v| [v.x, v.y]).collect();
        let mut inv = AntiInversion::new(&g, &x);
        for k in 0..g.contours.len() {
            assert!(!is_inverted(&g, &x, k), "contour {k}");
        }
        assert_eq!(inv.full_check(&g, &x), 0);
        assert!(inv.anchor.iter().all(|&a| a == 0));
        // Fold one vertex of the hole across the outer ring: its contour turns the wrong way.
        let mut y = x.clone();
        let v = g.contours[2].vertices[0];
        y[2 * v] = 0.5;
        y[2 * v + 1] = 0.5;
        assert!(inv.full_check(&g, &y) > 0);
    }
}
