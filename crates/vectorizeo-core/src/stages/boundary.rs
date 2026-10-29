//! Boundary tracer: turns a label map into one closed loop of
//! shared lattice vertices per segment, then removes redundant vertices on digital straight lines.
//!
//! Steps: vertex grid, tracing order, junction typing and parent lookup; vertex numbering;
//! per-segment loop tracing; vertex decimation with a staircase pattern table.

/// Vertex flag: removed by decimation.
const REMOVED: u8 = 0x40;
/// Vertex flag: locked against further decimation.
const LOCKED: u8 = 0x80;

/// Pixel offsets of the four pixels around a vertex, indexed by move direction.
const AX: [i32; 4] = [-1, -1, 0, 0];
const AY: [i32; 4] = [-1, 0, 0, -1];
/// Vertex moves: left, down, right, up.
const MX: [i32; 4] = [-1, 0, 1, 0];
const MY: [i32; 4] = [0, 1, 0, -1];
/// Pixel corners tried as start vertex.
const CX: [i32; 4] = [0, 1, 1, 0];
const CY: [i32; 4] = [0, 0, 1, 1];

/// Staircase patterns: (step bits, keep flags for len+1 vertices), MSB first.
const PATTERNS: [(&str, &str); 6] = [
    ("00000", "100001"),
    ("010101010", "1001001001"),
    ("010010010", "1001001001"),
    ("1000100010001", "10000100100001"),
    ("0010000100", "10000100001"),
    ("0101001010", "10000100001"),
];

struct Pattern {
    len: u32,
    bits: u32,
    keep: u32,
}

fn patterns() -> Vec<Pattern> {
    let parse = |s: &str| s.bytes().fold(0u32, |acc, b| acc << 1 | u32::from(b == b'1'));
    PATTERNS
        .iter()
        .map(|(bits, keep)| Pattern { len: bits.len() as u32, bits: parse(bits), keep: parse(keep) })
        .collect()
}

#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Vertex {
    pub x: f64,
    pub y: f64,
    /// Image corners and junctions of three or more segments. Never moved.
    pub fixed: bool,
    flags: u8,
}

/// One segment's outer boundary.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Contour {
    /// Indices into [`BoundaryGraph::vertices`], in tracing order (segment on the right-hand side).
    pub vertices: Vec<usize>,
    /// Label on the other side of the edge leaving each vertex (-1 = outside the image).
    pub edge_neighbors: Vec<i32>,
    /// Number of original pixel steps represented by the edge ending at each vertex.
    pub steps: Vec<u32>,
    /// Distinct neighbouring labels in first-seen order.
    pub neighbors: Vec<i32>,
    /// Enclosing segment, if this segment is a hole inside another one.
    pub parent: Option<usize>,
    /// Edge entry bit 0: vertex punctured as a sharp corner by the smoother.
    pub corner: Vec<bool>,
    /// Edge entry bit 3: the vertex's turn was the largest in its 5-vertex neighbourhood.
    pub turn_peak: Vec<bool>,
}

#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct BoundaryGraph {
    pub width: usize,
    pub height: usize,
    pub vertices: Vec<Vertex>,
    pub contours: Vec<Contour>,
    /// Surviving border vertices per side (left col, right col, top row, bottom row); they are
    /// numbered first, in that order.
    pub border_counts: [usize; 4],
}

struct Tracer<'a> {
    w: i32,
    h: i32,
    labels: &'a [u32],
    /// Lattice (W+1)×(H+1) → vertex index or -1.
    vidx: Vec<i32>,
    visited: Vec<i32>,
}

impl Tracer<'_> {
    fn label(&self, x: i32, y: i32) -> i32 {
        if x < 0 || y < 0 || x >= self.w || y >= self.h { -1 } else { self.labels[(y * self.w + x) as usize] as i32 }
    }

    fn lattice(&self, x: i32, y: i32) -> usize {
        (y * (self.w + 1) + x) as usize
    }

    /// Count boundary incidences per lattice point, then number vertices:
    /// left column, right column, top row, bottom row, then interior boundary points in raster order.
    fn number_vertices(&mut self) -> usize {
        let (w, h) = (self.w, self.h);
        let mut count = vec![0i32; ((w + 1) * (h + 1)) as usize];
        for y in 0..h - 1 {
            for x in 0..w - 1 {
                if self.label(x, y) != self.label(x, y + 1) {
                    count[self.lattice(x, y + 1)] += 1;
                    count[self.lattice(x + 1, y + 1)] += 1;
                }
                if self.label(x, y) != self.label(x + 1, y) {
                    count[self.lattice(x + 1, y)] += 1;
                    count[self.lattice(x + 1, y + 1)] += 1;
                }
            }
        }
        if self.label(w - 1, h - 1) != self.label(w - 2, h - 1) || self.label(w - 1, h - 1) != self.label(w - 1, h - 2)
        {
            let i = self.lattice(w - 1, h - 1);
            count[i] += 1;
        }
        let mut next = 0i32;
        let mut set = |v: &mut Vec<i32>, i: usize| {
            v[i] = next;
            next += 1;
        };
        let mut vidx = vec![-1i32; count.len()];
        for y in 0..=h {
            set(&mut vidx, self.lattice(0, y));
        }
        for y in 0..=h {
            set(&mut vidx, self.lattice(w, y));
        }
        for x in 1..w {
            set(&mut vidx, self.lattice(x, 0));
        }
        for x in 1..w {
            set(&mut vidx, self.lattice(x, h));
        }
        for y in 1..h {
            for x in 1..w {
                let i = self.lattice(x, y);
                if count[i] >= 1 {
                    set(&mut vidx, i);
                }
            }
        }
        self.vidx = vidx;
        next as usize
    }

    /// Walk the outer boundary of segment `k` from lattice point (x, y).
    fn trace(&mut self, k: i32, mut x: i32, mut y: i32) -> (Vec<usize>, Vec<i32>) {
        let mut verts = Vec::new();
        let mut nbrs = Vec::new();
        let mut prev_a: Option<(i32, i32)> = None;
        loop {
            verts.push(self.vidx[self.lattice(x, y)] as usize);
            // (next x, next y, direction, label B) of accepted moves.
            let mut cands: Vec<(i32, i32, usize, i32)> = Vec::new();
            let mut last_b = -1;
            for u in 0..4 {
                let (nx, ny) = (x + MX[u], y + MY[u]);
                if nx < 0 || nx > self.w || ny < 0 || ny > self.h {
                    continue;
                }
                let li = self.lattice(nx, ny);
                let a = self.label(x + AX[u], y + AY[u]);
                let b = self.label(x + AX[(u + 1) % 4], y + AY[(u + 1) % 4]);
                if self.vidx[li] >= 0 && a != b && a == k {
                    last_b = b;
                    if self.visited[li] != k {
                        cands.push((nx, ny, u, b));
                    }
                }
            }
            let chosen = if cands.len() < 2 {
                let here = self.lattice(x, y);
                self.visited[here] = k;
                match cands.first() {
                    Some(&c) => {
                        prev_a = Some((x + AX[c.2], y + AY[c.2]));
                        Some(c)
                    }
                    None => None,
                }
            } else if let Some(pa) = prev_a {
                // Checkerboard vertex: keep turning around the same pixel.
                cands.iter().copied().find(|c| (x + AX[c.2], y + AY[c.2]) == pa)
            } else {
                let c = cands[0];
                prev_a = Some((x + AX[c.2], y + AY[c.2]));
                Some(c)
            };
            match chosen {
                Some((nx, ny, _, b)) => {
                    nbrs.push(b);
                    x = nx;
                    y = ny;
                }
                // No move left: the closing edge back to the start; keep its neighbour label.
                None => nbrs.push(last_b),
            }
            if self.visited[self.lattice(x, y)] == k {
                return (verts, nbrs);
            }
        }
    }
}

/// Build the boundary graph for a label map (with vertex decimation when `decimate`).
impl BoundaryGraph {
    /// Even-odd crossing test of (x, y) against segment `s`'s own loop (holes
    /// are not tested).
    pub fn inside(&self, s: usize, x: f64, y: f64) -> bool {
        let vs = &self.contours[s].vertices;
        let mut inside = false;
        let Some(&last) = vs.last() else { return false };
        let mut b = &self.vertices[last];
        for &v in vs {
            let a = &self.vertices[v];
            if (a.y > y) != (b.y > y) && x < a.x + (y - a.y) * (b.x - a.x) / (b.y - a.y) {
                inside = !inside;
            }
            b = a;
        }
        inside
    }

    /// The shape under the image point (x, y)
    /// for Remove Background. The pixel's own segment if (x, y) lies in its loop, else the first
    /// untried label on growing square rings around the pixel whose loop contains the point.
    /// Quirk: after the ring's left column only the top and bottom pixels of the other
    /// columns are visited (the step is updated after each column), so the right side is
    /// never searched.
    pub fn shape_at(&self, labels: &[u32], x: f64, y: f64) -> Option<usize> {
        let (w, h) = (self.width as i32, self.height as i32);
        let (ix, iy) = (x as i32, y as i32);
        if ix < 0 || iy < 0 || ix >= w || iy >= h {
            return None;
        }
        let label = |px: i32, py: i32| labels[(py * w + px) as usize] as usize;
        let s = label(ix, iy);
        if self.inside(s, x, y) {
            return Some(s);
        }
        let mut tried = std::collections::BTreeSet::from([s]);
        let rings = (w - ix).max(ix).max((h - iy).max(iy));
        for r in 1..rings {
            let mut step = 1;
            for dx in -r..=r {
                let mut dy = -r;
                while dy <= r {
                    let (px, py) = (ix + dx, iy + dy);
                    if px >= 0 && px < w && py >= 0 && py < h {
                        let t = label(px, py);
                        if tried.insert(t) && self.inside(t, x, y) {
                            return Some(t);
                        }
                    }
                    dy += step;
                }
                step = if dx < r { 2 * r } else { 1 };
            }
        }
        None
    }
}

pub fn trace_boundaries(
    width: usize,
    height: usize,
    labels: &[u32],
    num_segments: usize,
    decimate: bool,
) -> BoundaryGraph {
    let (w, h) = (width as i32, height as i32);
    let mut t = Tracer { w, h, labels, vidx: Vec::new(), visited: vec![-1; ((w + 1) * (h + 1)) as usize] };
    let nverts = t.number_vertices();

    let mut vertices = vec![Vertex { x: 0.0, y: 0.0, fixed: false, flags: 0 }; nverts];
    for y in 0..=h {
        for x in 0..=w {
            let vi = t.vidx[t.lattice(x, y)];
            if vi >= 0 {
                // Image corners are fixed.
                let corner = (x == 0 || x == w) && (y == 0 || y == h);
                vertices[vi as usize] = Vertex { x: f64::from(x), y: f64::from(y), fixed: corner, flags: 0 };
            }
        }
    }

    // Trace each segment from the first boundary corner of its first pixel in raster order.
    let mut contours: Vec<Option<Contour>> = vec![None; num_segments];
    for y in 0..h {
        for x in 0..w {
            let k = t.label(x, y);
            if contours[k as usize].is_some() {
                continue;
            }
            if let Some(c) = (0..4).find(|&c| t.vidx[t.lattice(x + CX[c], y + CY[c])] >= 0) {
                let (verts, edge_neighbors) = t.trace(k, x + CX[c], y + CY[c]);
                let mut neighbors = Vec::new();
                for &n in &edge_neighbors {
                    if n >= 0 && !neighbors.contains(&n) {
                        neighbors.push(n);
                    }
                }
                let steps = vec![1; verts.len()];
                let n = verts.len();
                contours[k as usize] = Some(Contour {
                    vertices: verts,
                    edge_neighbors,
                    steps,
                    neighbors,
                    parent: None,
                    corner: vec![false; n],
                    turn_peak: vec![false; n],
                });
            }
        }
    }
    let mut contours: Vec<Contour> = contours.into_iter().map(|c| c.expect("every segment has a boundary")).collect();

    // Junctions: fewer than two equal pairs among the four pixels around a vertex.
    for y in 0..=h {
        for x in 0..=w {
            let vi = t.vidx[t.lattice(x, y)];
            if vi < 0 {
                continue;
            }
            let l = [0, 1, 2, 3].map(|u| t.label(x + AX[u], y + AY[u]));
            let pairs = u8::from(l[1] == l[2])
                + u8::from(l[2] == l[3])
                + u8::from(l[1] == l[3])
                + u8::from(l[0] == l[2])
                + u8::from(l[0] == l[1])
                + u8::from(l[0] == l[3]);
            if pairs < 2 {
                vertices[vi as usize].fixed = true;
            }
        }
    }

    let mut graph = BoundaryGraph { width, height, vertices, contours: Vec::new(), border_counts: [0; 4] };
    if decimate {
        decimate_vertices(&mut graph.vertices, &mut contours, w, h);
    }
    let side_len = [h + 1, h + 1, w - 1, w - 1];
    let mut start = 0usize;
    for (s, &len) in side_len.iter().enumerate() {
        let range = start..start + len as usize;
        graph.border_counts[s] = graph.vertices[range].iter().filter(|v| v.flags & REMOVED == 0).count();
        start += len as usize;
    }
    compact(&mut graph.vertices, &mut contours);

    // Parent: the first neighbour whose own outer boundary does not touch this segment.
    let parents: Vec<Option<usize>> = (0..contours.len())
        .map(|i| {
            contours[i]
                .neighbors
                .iter()
                .find(|&&j| !contours[j as usize].neighbors.contains(&(i as i32)))
                .map(|&j| j as usize)
        })
        .collect();
    for (c, p) in contours.iter_mut().zip(parents) {
        c.parent = p;
    }
    graph.contours = contours;
    graph
}

/// First half: drop non-junction image-border vertices and straight staircase interiors.
fn decimate_vertices(verts: &mut [Vertex], contours: &mut [Contour], w: i32, h: i32) {
    // Border vertices are numbered first: left col (h+1), right col (h+1), top row (w-1), bottom row (w-1).
    let sides = [h + 1, h + 1, w - 1, w - 1];
    let mut idx = 0usize;
    for (s, &len) in sides.iter().enumerate() {
        let skip_ends = s < 2;
        if skip_ends {
            idx += 1;
        }
        let inner = if skip_ends { len - 2 } else { len };
        for _ in 0..inner {
            if !verts[idx].fixed {
                verts[idx].flags |= REMOVED;
            }
            idx += 1;
        }
        if skip_ends {
            idx += 1;
        }
    }
    for v in verts.iter_mut().take(((w + h) * 2) as usize) {
        v.flags |= LOCKED;
    }

    let pats = patterns();
    for c in contours.iter() {
        let n = c.vertices.len();
        if n < 20 {
            continue;
        }
        let (mut run, mut last_dx, mut last_dy, mut xb, mut yb) = (0u32, 0i32, 0i32, 0u32, 0u32);
        for i in 1..n + 31 {
            let pi = c.vertices[(i - 1) % n];
            let ci = c.vertices[i % n];
            let ok =
                verts[pi].flags & REMOVED == 0 && verts[ci].flags & LOCKED == 0 && !verts[pi].fixed && !verts[ci].fixed;
            if !ok {
                (run, xb, yb, last_dx, last_dy) = (0, 0, 0, 0, 0);
                continue;
            }
            if i > 30 {
                verts[pi].flags |= LOCKED;
            }
            let dx = (verts[pi].x - verts[ci].x + 0.5).floor() as i32;
            let dy = (verts[pi].y - verts[ci].y + 0.5).floor() as i32;
            if (last_dx != 0 && dx != 0 && dx != last_dx) || (last_dy != 0 && dy != 0 && dy != last_dy) {
                (run, yb, xb) = (0, 0, 0);
            }
            if dx != 0 {
                last_dx = dx;
            }
            if dy != 0 {
                last_dy = dy;
            }
            xb = u32::from(dx != 0) | xb << 1;
            run += 1;
            yb = u32::from(dy != 0) | yb << 1;
            if run >= 32 {
                run = 31;
            } else if run < 5 {
                continue;
            }
            // First pattern whose length fits and whose bits match either axis.
            let hit = pats.iter().find(|p| {
                let mask = (1u32 << p.len) - 1;
                p.len <= run && ((mask & xb) == p.bits || (mask & yb) == p.bits)
            });
            if let Some(p) = hit {
                let mut keep = p.keep;
                let mut at = i + n;
                for _ in 0..=p.len {
                    if keep & 1 == 0 {
                        verts[c.vertices[at % n]].flags |= REMOVED | LOCKED;
                    }
                    at -= 1;
                    keep >>= 1;
                }
                (run, yb, xb, last_dy, last_dx) = (0, 0, 0, 0, 0);
            }
        }
    }
}

/// Second half: renumber surviving vertices and shorten every contour,
/// recording how many original steps each surviving edge covers.
fn compact(verts: &mut Vec<Vertex>, contours: &mut [Contour]) {
    let mut remap = vec![usize::MAX; verts.len()];
    let mut kept = Vec::with_capacity(verts.len());
    for (i, v) in verts.iter().enumerate() {
        if v.flags & REMOVED == 0 {
            remap[i] = kept.len();
            kept.push(v.clone());
        }
    }
    *verts = kept;
    for c in contours.iter_mut() {
        let n = c.vertices.len();
        let (mut vs, mut ns, mut steps) = (Vec::new(), Vec::new(), Vec::new());
        let mut last = 0usize;
        for i in 0..n {
            let r = remap[c.vertices[i]];
            if r != usize::MAX {
                vs.push(r);
                ns.push(c.edge_neighbors[i]);
                steps.push((i - last) as u32);
                last = i;
            }
        }
        if let Some(first) = steps.first_mut() {
            *first = (n - last) as u32;
        }
        c.corner = vec![false; vs.len()];
        c.turn_peak = vec![false; vs.len()];
        c.vertices = vs;
        c.edge_neighbors = ns;
        c.steps = steps;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn labels_from(rows: &[&str]) -> (usize, usize, Vec<u32>) {
        let h = rows.len();
        let w = rows[0].len();
        let l = rows.iter().flat_map(|r| r.bytes().map(|b| u32::from(b - b'0'))).collect();
        (w, h, l)
    }

    #[test]
    fn square_inside_square() {
        let (w, h, l) = labels_from(&["0000", "0110", "0110", "0000"]);
        let g = trace_boundaries(w, h, &l, 2, false);
        assert_eq!(g.contours[1].vertices.len(), 8);
        assert_eq!(g.contours[1].parent, Some(0));
        assert_eq!(g.contours[0].parent, None);
        // Outer segment's loop runs around the image border.
        assert_eq!(g.contours[0].vertices.len(), 16);
    }

    /// The pixel's own segment wins; the parent's loop (holes ignored) contains the child too.
    #[test]
    fn shape_at_prefers_the_pixel_segment() {
        let (w, h, l) = labels_from(&["0000", "0110", "0110", "0000"]);
        let g = trace_boundaries(w, h, &l, 2, false);
        assert_eq!(g.shape_at(&l, 1.5, 1.5), Some(1));
        assert_eq!(g.shape_at(&l, 0.5, 3.5), Some(0));
        assert!(g.inside(0, 1.5, 1.5));
        assert_eq!(g.shape_at(&l, -1.5, 1.0), None);
        assert_eq!(g.shape_at(&l, 4.0, 1.0), None);
    }

    #[test]
    fn decimation_keeps_shape_corners() {
        // A 30x30 square in a 40x40 image: decimation must keep its 4 corners.
        let mut l = vec![0u32; 40 * 40];
        for y in 5..35 {
            for x in 5..35 {
                l[y * 40 + x] = 1;
            }
        }
        let g = trace_boundaries(40, 40, &l, 2, true);
        let c = &g.contours[1];
        let pts: Vec<(f64, f64)> = c.vertices.iter().map(|&v| (g.vertices[v].x, g.vertices[v].y)).collect();
        for corner in [(5.0, 5.0), (35.0, 5.0), (35.0, 35.0), (5.0, 35.0)] {
            assert!(pts.contains(&corner), "missing {corner:?} in {pts:?}");
        }
        assert!(pts.len() < 120, "expected decimation, got {}", pts.len());
        assert_eq!(c.steps.iter().sum::<u32>(), 120);
    }
}
