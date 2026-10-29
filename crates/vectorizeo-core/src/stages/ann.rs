//! The k-nearest-neighbour search the engine links in: ANN 1.1 ("ANN, Copyright 1997-2005
//! University of Maryland and Sunil Arya and David Mount"), `ANNkd_tree` built with bucket size 1
//! and `ANN_KD_SUGGEST` (sliding-midpoint splits), queried with `annkSearch`. Reproduced to the
//! traversal order because, among points at equal distance, the first one the search reaches
//! wins the last slots.

enum Node {
    /// Empty bucket (`KD_TRIVIAL`).
    Trivial,
    /// One point (bucket size 1).
    Leaf(usize),
    Split {
        dim: usize,
        cut: f64,
        lo_bound: f64,
        hi_bound: f64,
        lo: Box<Node>,
        hi: Box<Node>,
    },
}

pub(crate) struct KdTree<'p> {
    pts: &'p [[f64; 2]],
    root: Node,
    lo: [f64; 2],
    hi: [f64; 2],
}

/// `annEnclRect`.
fn enclosing_rect(pts: &[[f64; 2]], idx: &[usize]) -> ([f64; 2], [f64; 2]) {
    let mut lo = pts[idx[0]];
    let mut hi = lo;
    for &i in &idx[1..] {
        for d in 0..2 {
            let c = pts[i][d];
            if c < lo[d] {
                lo[d] = c;
            } else if c > hi[d] {
                hi[d] = c;
            }
        }
    }
    (lo, hi)
}

/// `annSpread` / `annMinMax`.
fn min_max(pts: &[[f64; 2]], idx: &[usize], d: usize) -> (f64, f64) {
    let mut min = pts[idx[0]][d];
    let mut max = min;
    for &i in &idx[1..] {
        let c = pts[i][d];
        if c < min {
            min = c;
        } else if c > max {
            max = c;
        }
    }
    (min, max)
}

/// `annPlaneSplit`: permute `idx` into < cv | == cv | > cv and return the two break points.
fn plane_split(pts: &[[f64; 2]], idx: &mut [usize], d: usize, cv: f64) -> (usize, usize) {
    let n = idx.len() as isize;
    let pa = |idx: &[usize], i: isize| pts[idx[i as usize]][d];
    let (mut l, mut r) = (0isize, n - 1);
    loop {
        while l < n && pa(idx, l) < cv {
            l += 1;
        }
        while r >= 0 && pa(idx, r) >= cv {
            r -= 1;
        }
        if l > r {
            break;
        }
        idx.swap(l as usize, r as usize);
        l += 1;
        r -= 1;
    }
    let br1 = l;
    r = n - 1;
    loop {
        while l < n && pa(idx, l) <= cv {
            l += 1;
        }
        while r >= br1 && pa(idx, r) > cv {
            r -= 1;
        }
        if l > r {
            break;
        }
        idx.swap(l as usize, r as usize);
        l += 1;
        r -= 1;
    }
    (br1 as usize, l as usize)
}

/// `sl_midpt_split`: cut the longest side of the box at its middle, slid onto the points if the
/// middle misses them. Returns (dimension, cut value, number of points on the low side).
fn sl_midpt_split(pts: &[[f64; 2]], idx: &mut [usize], lo: &[f64; 2], hi: &[f64; 2]) -> (usize, f64, usize) {
    const ERR: f64 = 0.001;
    let n = idx.len();
    let max_length = (hi[0] - lo[0]).max(hi[1] - lo[1]);
    let mut max_spread = -1.0;
    let mut cut_dim = 0;
    for d in 0..2 {
        if hi[d] - lo[d] >= (1.0 - ERR) * max_length {
            let (min, max) = min_max(pts, idx, d);
            if max - min > max_spread {
                max_spread = max - min;
                cut_dim = d;
            }
        }
    }
    let ideal = (lo[cut_dim] + hi[cut_dim]) / 2.0;
    let (min, max) = min_max(pts, idx, cut_dim);
    let cut = if ideal < min {
        min
    } else if ideal > max {
        max
    } else {
        ideal
    };
    let (br1, br2) = plane_split(pts, idx, cut_dim, cut);
    let n_lo = if ideal < min {
        1
    } else if ideal > max {
        n - 1
    } else if br1 > n / 2 {
        br1
    } else if br2 < n / 2 {
        br2
    } else {
        n / 2
    };
    (cut_dim, cut, n_lo)
}

/// `rkd_tree` with bucket size 1.
fn build(pts: &[[f64; 2]], idx: &mut [usize], lo: &mut [f64; 2], hi: &mut [f64; 2]) -> Node {
    match idx.len() {
        0 => return Node::Trivial,
        1 => return Node::Leaf(idx[0]),
        _ => {}
    }
    let (dim, cut, n_lo) = sl_midpt_split(pts, idx, lo, hi);
    let (lv, hv) = (lo[dim], hi[dim]);
    let (a, b) = idx.split_at_mut(n_lo);
    hi[dim] = cut;
    let lo_node = build(pts, a, lo, hi);
    hi[dim] = hv;
    lo[dim] = cut;
    let hi_node = build(pts, b, lo, hi);
    lo[dim] = lv;
    Node::Split { dim, cut, lo_bound: lv, hi_bound: hv, lo: Box::new(lo_node), hi: Box::new(hi_node) }
}

/// `ANNmin_k`: the k smallest (distance, point) pairs; equal keys keep insertion order.
struct MinK {
    k: usize,
    items: Vec<(f64, usize)>,
}

impl MinK {
    fn max_key(&self) -> f64 {
        if self.items.len() == self.k { self.items[self.k - 1].0 } else { f64::MAX }
    }

    fn insert(&mut self, key: f64, info: usize) {
        let mut i = self.items.len();
        while i > 0 && self.items[i - 1].0 > key {
            i -= 1;
        }
        self.items.insert(i, (key, info));
        self.items.truncate(self.k);
    }
}

struct Search<'a> {
    pts: &'a [[f64; 2]],
    q: [f64; 2],
    max_err: f64,
    mk: MinK,
}

impl Search<'_> {
    fn visit(&mut self, node: &Node, box_dist: f64) {
        match node {
            Node::Trivial => {}
            Node::Leaf(i) => {
                let min_dist = self.mk.max_key();
                let p = self.pts[*i];
                let t0 = self.q[0] - p[0];
                let d0 = t0 * t0;
                if d0 > min_dist {
                    return;
                }
                let t1 = self.q[1] - p[1];
                let dist = d0 + t1 * t1;
                if dist > min_dist {
                    return;
                }
                self.mk.insert(dist, *i);
            }
            Node::Split { dim, cut, lo_bound, hi_bound, lo, hi } => {
                let cut_diff = self.q[*dim] - cut;
                let (near, far, box_diff) =
                    if cut_diff < 0.0 { (lo, hi, lo_bound - self.q[*dim]) } else { (hi, lo, self.q[*dim] - hi_bound) };
                self.visit(near, box_dist);
                let box_diff = box_diff.max(0.0);
                let box_dist = box_dist + (cut_diff * cut_diff - box_diff * box_diff);
                if box_dist * self.max_err < self.mk.max_key() {
                    self.visit(far, box_dist);
                }
            }
        }
    }
}

impl<'p> KdTree<'p> {
    /// `ANNkd_tree(pts, n, 2, 1, ANN_KD_SUGGEST)`.
    pub(crate) fn new(pts: &'p [[f64; 2]]) -> Self {
        let mut idx: Vec<usize> = (0..pts.len()).collect();
        if pts.is_empty() {
            return Self { pts, root: Node::Trivial, lo: [0.0; 2], hi: [0.0; 2] };
        }
        let (lo, hi) = enclosing_rect(pts, &idx);
        let root = build(pts, &mut idx, &mut lo.clone(), &mut hi.clone());
        Self { pts, root, lo, hi }
    }

    /// `annkSearch(q, k, idx, dists, eps)`: the k nearest points with their squared distances.
    pub(crate) fn search(&self, q: [f64; 2], k: usize, eps: f64) -> Vec<(f64, usize)> {
        // annBoxDistance
        let mut box_dist = 0.0;
        for d in 0..2 {
            let t = if q[d] < self.lo[d] {
                self.lo[d] - q[d]
            } else if q[d] > self.hi[d] {
                q[d] - self.hi[d]
            } else {
                0.0
            };
            box_dist += t * t;
        }
        let e = 1.0 + eps;
        let mut s = Search { pts: self.pts, q, max_err: e * e, mk: MinK { k, items: Vec::with_capacity(k + 1) } };
        s.visit(&self.root, box_dist);
        s.mk.items
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finds_the_nearest_points() {
        let pts: Vec<[f64; 2]> = (0..100).map(|i| [f64::from(i % 10), f64::from(i / 10)]).collect();
        let tree = KdTree::new(&pts);
        let q = [3.2, 4.9];
        let got = tree.search(q, 5, 0.0);
        let mut all: Vec<(f64, usize)> =
            pts.iter().enumerate().map(|(i, p)| ((p[0] - q[0]).powi(2) + (p[1] - q[1]).powi(2), i)).collect();
        all.sort_by(|a, b| a.0.total_cmp(&b.0));
        let want: Vec<f64> = all[..5].iter().map(|a| a.0).collect();
        assert_eq!(got.iter().map(|a| a.0).collect::<Vec<_>>(), want);
    }

    #[test]
    fn duplicate_points_terminate() {
        let pts = vec![[1.0, 1.0]; 7];
        let tree = KdTree::new(&pts);
        assert_eq!(tree.search([1.0, 1.0], 3, 0.0).len(), 3);
    }
}
