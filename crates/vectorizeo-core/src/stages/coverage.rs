//! Exact pixel coverage of contour polygons, the geometry behind measurement type 1 of the
//! contour smoother.
//!
//! Every edge is clipped against the pixel grid, closes the pieces inside each pixel along
//! the pixel border and takes the shoelace area of each fragment; its gradient
//! slides the border crossing points along their grid lines. Here the same quantities come from
//! `area = ∮ x dy` accumulated per cell: a piece in cell (c, r) contributes `dy·(x̄ − c)` there and
//! `dy` to every cell left of it in the same row. Pieces lying on a grid line belong to the cell
//! right of / below it, which is the cell they are filed under.

use std::collections::BTreeMap;

/// A positive-length piece of one contour edge inside one pixel.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Piece {
    /// Index of the edge's start vertex within the contour.
    pub edge: usize,
    /// Parameter range of the piece along the edge (0 at the start vertex).
    pub t0: f64,
    pub t1: f64,
}

/// One contour's coverage of one pixel.
#[derive(Clone, Debug, Default)]
pub(crate) struct Cell {
    /// Signed area of the contour polygon inside the pixel.
    pub area: f64,
    pub pieces: Vec<Piece>,
}

/// Split the edge a→b at the integer grid lines and report each piece with its cell
/// (column, row) and parameter range.
fn split_edge(a: [f64; 2], b: [f64; 2], mut f: impl FnMut(i64, i64, [f64; 2], [f64; 2], f64, f64)) {
    let d = [b[0] - a[0], b[1] - a[1]];
    // Parameters of the grid crossings, with the exact crossing coordinate where known.
    let mut cuts: Vec<(f64, [Option<f64>; 2])> = vec![(0.0, [None, None]), (1.0, [None, None])];
    for axis in 0..2 {
        if d[axis] == 0.0 {
            continue;
        }
        let (lo, hi) = (a[axis].min(b[axis]), a[axis].max(b[axis]));
        let mut g = lo.floor() + 1.0;
        while g < hi {
            let mut fix = [None, None];
            fix[axis] = Some(g);
            cuts.push(((g - a[axis]) / d[axis], fix));
            g += 1.0;
        }
    }
    cuts.sort_by(|p, q| p.0.total_cmp(&q.0));
    let at = |t: f64, fix: [Option<f64>; 2]| -> [f64; 2] {
        if t == 0.0 {
            return a;
        }
        if t == 1.0 {
            return b;
        }
        [fix[0].unwrap_or(a[0] + t * d[0]), fix[1].unwrap_or(a[1] + t * d[1])]
    };
    for w in cuts.windows(2) {
        let (t0, t1) = (w[0].0, w[1].0);
        if t1 <= t0 {
            continue;
        }
        let (p0, p1) = (at(t0, w[0].1), at(t1, w[1].1));
        let mid = [(p0[0] + p1[0]) * 0.5, (p0[1] + p1[1]) * 0.5];
        f(mid[0].floor() as i64, mid[1].floor() as i64, p0, p1, t0, t1);
    }
}

/// Coverage of one closed polygon: every pixel (column, row) that one of its edges passes
/// through, with the polygon's area inside it. Cells outside the image are included because
/// they carry area to the cells on their left.
pub(crate) fn polygon_cells(pts: &[[f64; 2]]) -> BTreeMap<(i64, i64), Cell> {
    // (row, col) → (Σ dy·(x̄ − col), Σ dy, pieces)
    let mut rows: BTreeMap<(i64, i64), (f64, f64, Vec<Piece>)> = BTreeMap::new();
    let n = pts.len();
    for i in 0..n {
        let (a, b) = (pts[i], pts[(i + 1) % n]);
        split_edge(a, b, |c, r, p0, p1, t0, t1| {
            let dy = p1[1] - p0[1];
            let e = rows.entry((r, c)).or_default();
            e.0 += dy * ((p0[0] + p1[0]) * 0.5 - c as f64);
            e.1 += dy;
            e.2.push(Piece { edge: i, t0, t1 });
        });
    }
    let mut out = BTreeMap::new();
    let mut carry = 0.0;
    let mut row = i64::MIN;
    // Walk each row from the right so `carry` is the Σ dy of the cells right of the current one.
    for (&(r, c), (local, dy, pieces)) in rows.iter_mut().rev() {
        if r != row {
            row = r;
            carry = 0.0;
        }
        out.insert((c, r), Cell { area: *local + carry, pieces: std::mem::take(pieces) });
        carry += *dy;
    }
    out
}

/// ∂area/∂(start, end) of an edge a→b restricted to the parameter range [t0, t1]
/// (`∫ ((1−t)·δa + t·δb) · (dy, −dx) dt`).
pub(crate) fn piece_gradient(a: [f64; 2], b: [f64; 2], t0: f64, t1: f64) -> ([f64; 2], [f64; 2]) {
    let nrm = [b[1] - a[1], a[0] - b[0]];
    let len = t1 - t0;
    let sq = (t1 * t1 - t0 * t0) * 0.5;
    let wa = len - sq;
    ([nrm[0] * wa, nrm[1] * wa], [nrm[0] * sq, nrm[1] * sq])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn square_on_the_grid_covers_whole_pixels() {
        // 2×2 block of pixels, positively oriented (∮x dy > 0: right side goes down).
        let sq = [[1.0, 1.0], [3.0, 1.0], [3.0, 3.0], [1.0, 3.0]];
        let cells = polygon_cells(&sq);
        // Edges on grid lines belong to the pixel right of / below them, so (2,2) has no edge
        // and the cells right of and below the square get the outer edges with zero area.
        for (c, r) in [(1, 1), (2, 1), (1, 2)] {
            assert!((cells[&(c, r)].area - 1.0).abs() < 1e-12, "{:?}", cells[&(c, r)]);
        }
        assert!(!cells.contains_key(&(2, 2)));
        for (c, r) in [(3, 1), (3, 2), (1, 3), (2, 3)] {
            assert!(cells[&(c, r)].area.abs() < 1e-12, "{:?}", cells[&(c, r)]);
        }
    }

    #[test]
    fn diagonal_cut_and_its_gradient() {
        // Triangle (0,0) (2,0) (0,2): pixel (0,0) is fully inside, pixels (1,0) and (0,1) half.
        let tri = [[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]];
        let cells = polygon_cells(&tri);
        assert!((cells[&(0, 0)].area - 1.0).abs() < 1e-12);
        assert!((cells[&(1, 0)].area - 0.5).abs() < 1e-12);
        assert!((cells[&(0, 1)].area - 0.5).abs() < 1e-12);
        // Moving (2,0) right by h grows pixel (1,0)'s area; compare with a finite difference.
        let h = 1e-6;
        let moved = polygon_cells(&[[0.0, 0.0], [2.0 + h, 0.0], [0.0, 2.0]]);
        let fd = (moved[&(1, 0)].area - cells[&(1, 0)].area) / h;
        let mut g = 0.0;
        for p in &cells[&(1, 0)].pieces {
            let (a, b) = (tri[p.edge], tri[(p.edge + 1) % 3]);
            let (ga, gb) = piece_gradient(a, b, p.t0, p.t1);
            if p.edge == 1 {
                g += ga[0];
            }
            if p.edge == 0 {
                g += gb[0];
            }
        }
        assert!((fd - g).abs() < 1e-4, "fd {fd} analytic {g}");
    }
}
