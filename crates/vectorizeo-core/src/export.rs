//! Vector export (driving the SVG writer), built on the output of
//! Bezier fitting.
//!
//! Every segment is one path: its own contour, then — in the cut-out modes — one hole loop per
//! ring of child pieces that border it, walked backwards. With colour grouping (the default)
//! segments of identical colour are wrapped in `<g id="#rrggbbaa">`.

use std::collections::HashMap;

use crate::eps::EpsWriter;
use crate::stages::bezier::{BezierPaths, Element};
use crate::stages::boundary::BoundaryGraph;
use crate::svg::{P, SvgWriter};

/// Output file format.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Format {
    Svg,
    /// Encapsulated PostScript (our own writer, see [`crate::eps`]).
    Eps,
}

impl Format {
    /// From a file extension (case-insensitive): `eps` is EPS, anything else SVG.
    pub fn from_extension(ext: &str) -> Self {
        if ext.eq_ignore_ascii_case("eps") { Format::Eps } else { Format::Svg }
    }

    pub fn extension(self) -> &'static str {
        match self {
            Format::Svg => "svg",
            Format::Eps => "eps",
        }
    }
}

/// What the export walk drives: one call per element, in the order the SVG writer needs.
pub(crate) trait VectorWriter {
    fn group_open(&mut self, c: [u8; 4]);
    fn group_close(&mut self);
    fn path_open(&mut self, c: [u8; 4]);
    fn path_close(&mut self);
    /// `first`: the first element of a loop (outer or hole).
    fn line(&mut self, a: P, b: P, first: bool);
    fn cubic(&mut self, c: &[P; 4], first: bool);
    fn finish(self) -> String;
}

impl VectorWriter for SvgWriter {
    fn group_open(&mut self, c: [u8; 4]) {
        SvgWriter::group_open(self, c);
    }
    fn group_close(&mut self) {
        SvgWriter::group_close(self);
    }
    fn path_open(&mut self, c: [u8; 4]) {
        SvgWriter::path_open(self, c);
    }
    fn path_close(&mut self) {
        SvgWriter::path_close(self);
    }
    fn line(&mut self, a: P, b: P, first: bool) {
        SvgWriter::line(self, a, b, first);
    }
    fn cubic(&mut self, c: &[P; 4], first: bool) {
        SvgWriter::cubic(self, c, first);
    }
    fn finish(self) -> String {
        SvgWriter::finish(self)
    }
}

/// Export option: how shapes enclosed by other shapes are written (GUI "Shape mode").
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ShapeMode {
    /// 0: "Stack shapes on top of each other".
    Stacked,
    /// 1: "Place shapes in cut-outs in shapes below".
    CutOuts,
    /// 2: "Place shapes in cut-outs and group by color" (default).
    CutOutsGrouped,
}

/// Export options (defaults: mode 2, stroking mode 1 = no stroke).
#[derive(Clone, Copy, Debug)]
pub struct ExportOptions {
    pub mode: ShapeMode,
    /// `shape_stroking_mode == 0`: hairline stroke in the fill colour.
    pub stroke: bool,
}

impl Default for ExportOptions {
    fn default() -> Self {
        Self { mode: ShapeMode::CutOutsGrouped, stroke: false }
    }
}

/// Path entry of a segment record: kind 0 = line over contour edge `index`,
/// 1 = cubic `index` as fitted, 2 = cubic stored reversed.
#[derive(Clone, Copy, Debug)]
struct Entry {
    kind: u8,
    index: usize,
    /// Segment on the other side.
    neighbour: i32,
    from: usize,
    to: usize,
}

/// Entries of every contour, from the Bezier elements (or one line per edge for polygons).
fn entries(g: &BoundaryGraph, bz: Option<&BezierPaths>) -> Vec<Vec<Entry>> {
    g.contours
        .iter()
        .enumerate()
        .map(|(k, c)| {
            let n = c.vertices.len();
            let Some(bz) = bz else {
                return (0..n)
                    .map(|i| Entry {
                        kind: 0,
                        index: i,
                        neighbour: c.edge_neighbors[i],
                        from: c.vertices[i],
                        to: c.vertices[(i + 1) % n],
                    })
                    .collect();
            };
            // takes the neighbour from the edge leaving the line's start vertex or
            // the cubic's first sample.
            bz.paths[k]
                .iter()
                .map(|e| match *e {
                    Element::Line { at, from, to } => {
                        Entry { kind: 0, index: at, neighbour: c.edge_neighbors[at], from, to }
                    }
                    Element::Cubic { at, from, to, fit, reversed } => Entry {
                        kind: if reversed { 2 } else { 1 },
                        index: fit,
                        neighbour: c.edge_neighbors[(at + 1) % n],
                        from,
                        to,
                    },
                })
                .collect()
        })
        .collect()
}

/// Hole loops per segment as lists of (contour, entry index).
fn holes(g: &BoundaryGraph, ents: &[Vec<Entry>]) -> Vec<Vec<Vec<(usize, usize)>>> {
    let n = g.contours.len();
    let mut children: Vec<Vec<usize>> = vec![Vec::new(); n];
    for (k, c) in g.contours.iter().enumerate() {
        if let Some(p) = c.parent {
            children[p].push(k);
        }
    }
    let mut out = vec![Vec::new(); n];
    for s in 0..n {
        // Runs of child entries bordering s.
        let mut pieces: Vec<Vec<(usize, usize)>> = Vec::new();
        for &c in &children[s] {
            let mut run = Vec::new();
            for (i, e) in ents[c].iter().enumerate() {
                if e.neighbour == s as i32 {
                    run.push((c, i));
                } else if !run.is_empty() {
                    pieces.push(std::mem::take(&mut run));
                }
            }
            if !run.is_empty() {
                pieces.push(run);
            }
        }
        // next[i] = the last piece starting where piece i ends.
        let ent = |&(c, i): &(usize, usize)| ents[c][i];
        let mut next: Vec<isize> = vec![-1; pieces.len()];
        for (i, pi) in pieces.iter().enumerate() {
            let end = ent(pi.last().expect("non-empty piece")).to;
            for (j, pj) in pieces.iter().enumerate() {
                if ent(&pj[0]).from == end {
                    next[i] = j as isize;
                }
            }
        }
        for start in 0..pieces.len() {
            if next[start] < 0 {
                continue;
            }
            let mut ring = Vec::new();
            let mut cur = start;
            while next[cur] >= 0 {
                ring.extend_from_slice(&pieces[cur]);
                let nx = next[cur] as usize;
                next[cur] = -1;
                cur = nx;
            }
            out[s].push(ring);
        }
    }
    out
}

/// Segments grouped by identical colour, groups in first-seen order. Each entry
/// carries whether it opens and/or closes its group.
fn colour_groups(colors: &[[u8; 4]]) -> Vec<(usize, bool, bool)> {
    let mut index: HashMap<[u8; 4], usize> = HashMap::new();
    let mut groups: Vec<Vec<usize>> = Vec::new();
    for (k, &c) in colors.iter().enumerate() {
        let gi = *index.entry(c).or_insert_with(|| {
            groups.push(Vec::new());
            groups.len() - 1
        });
        groups[gi].push(k);
    }
    let mut order = Vec::with_capacity(colors.len());
    for gr in groups {
        let last = gr.len() - 1;
        for (i, k) in gr.into_iter().enumerate() {
            order.push((k, i == 0, i == last));
        }
    }
    order
}

/// ViewBox = largest vertex coordinates rounded up from .5 (truncating
/// `x + 0.5`); size in pt = viewBox · 72 / 72 in single precision, truncated.
fn extent(g: &BoundaryGraph) -> ([i32; 2], [i32; 2]) {
    let mut view = [0i32; 2];
    for v in &g.vertices {
        for (d, c) in [v.x, v.y].into_iter().enumerate() {
            if f64::from(view[d]) < c {
                view[d] = (c + 0.5) as i32;
            }
        }
    }
    const DPI: f32 = 72.0; // output object +0 (ctor)
    let scale = 1.0f32 / DPI;
    let size = view.map(|v| (scale * (72.0f32 * v as f32)) as i32);
    (view, size)
}

/// Write the SVG document for the smoothed contours (`bz` = Bezier fit, or `None` to write the
/// polygons) with each segment's colour.
pub fn export_svg(g: &BoundaryGraph, bz: Option<&BezierPaths>, colors: &[[u8; 4]], opts: ExportOptions) -> String {
    export_svg_hidden(g, bz, colors, opts, &[])
}

/// [`export_svg`] without the segments marked in `hidden` (Remove Background; indices past its end
/// are visible). Only the hidden
/// segment's own path is skipped: its colour group is still opened and closed, and parents keep
/// the hole loops around it.
pub fn export_svg_hidden(
    g: &BoundaryGraph,
    bz: Option<&BezierPaths>,
    colors: &[[u8; 4]],
    opts: ExportOptions,
    hidden: &[bool],
) -> String {
    export(g, bz, colors, opts, hidden, Format::Svg)
}

/// Write the document in `format`; see [`export_svg_hidden`] for `hidden`. EPS has the same
/// shapes, order and holes as the SVG.
pub fn export(
    g: &BoundaryGraph,
    bz: Option<&BezierPaths>,
    colors: &[[u8; 4]],
    opts: ExportOptions,
    hidden: &[bool],
    format: Format,
) -> String {
    let (view, size) = extent(g);
    match format {
        Format::Svg => write_shapes(SvgWriter::header(view, size, opts.stroke), g, bz, colors, opts, hidden),
        Format::Eps => write_shapes(EpsWriter::header(view, size, opts.stroke), g, bz, colors, opts, hidden),
    }
}

fn write_shapes<W: VectorWriter>(
    mut w: W,
    g: &BoundaryGraph,
    bz: Option<&BezierPaths>,
    colors: &[[u8; 4]],
    opts: ExportOptions,
    hidden: &[bool],
) -> String {
    let ents = entries(g, bz);
    let cut_outs = opts.mode != ShapeMode::Stacked;
    let holes = if cut_outs { holes(g, &ents) } else { Vec::new() };
    let order: Vec<(usize, bool, bool)> = if opts.mode == ShapeMode::CutOutsGrouped {
        colour_groups(colors)
    } else {
        (0..colors.len()).map(|k| (k, false, false)).collect()
    };
    let pos = |v: usize| -> P { [g.vertices[v].x, g.vertices[v].y] };
    let emit = |w: &mut W, c: usize, e: Entry, first: bool, reversed: bool| {
        let kind = match (e.kind, reversed) {
            (1, true) => 2,
            (2, true) => 1,
            (k, _) => k,
        };
        match kind {
            0 => {
                let vs = &g.contours[c].vertices;
                let (a, b) = (pos(vs[e.index]), pos(vs[(e.index + 1) % vs.len()]));
                if reversed { w.line(b, a, first) } else { w.line(a, b, first) }
            }
            _ => {
                let f = bz.expect("cubic entries come from the Bezier fit").fits[e.index];
                let pts = if kind == 1 { f } else { [f[3], f[2], f[1], f[0]] };
                w.cubic(&pts, first);
            }
        }
    };
    for (s, open, close) in order {
        let color = colors[s];
        if !SvgWriter::visible(color) {
            continue;
        }
        if open {
            w.group_open(color);
        }
        if hidden.get(s).copied().unwrap_or(false) {
            if close {
                w.group_close();
            }
            continue;
        }
        w.path_open(color);
        for (i, &e) in ents[s].iter().enumerate() {
            emit(&mut w, s, e, i == 0, false);
        }
        if cut_outs {
            for ring in &holes[s] {
                // The ring is walked backwards.
                for (i, &(c, j)) in ring.iter().rev().enumerate() {
                    emit(&mut w, c, ents[c][j], i == 0, true);
                }
            }
        }
        w.path_close();
        if close {
            w.group_close();
        }
    }
    w.finish()
}

/// A segment's outline as the GUI draws it (always cut-out style): its own loop,
/// then each hole loop, as polylines with every cubic flattened into `steps` pieces.
pub fn outline(g: &BoundaryGraph, bz: Option<&BezierPaths>, s: usize, steps: usize) -> Vec<Vec<P>> {
    let ents = entries(g, bz);
    let pos = |v: usize| -> P { [g.vertices[v].x, g.vertices[v].y] };
    let push = |out: &mut Vec<P>, c: usize, e: Entry, reversed: bool| {
        let kind = match (e.kind, reversed) {
            (1, true) => 2,
            (2, true) => 1,
            (k, _) => k,
        };
        let pts: Vec<P> = match kind {
            0 => {
                let vs = &g.contours[c].vertices;
                let (a, b) = (pos(vs[e.index]), pos(vs[(e.index + 1) % vs.len()]));
                if reversed { vec![b, a] } else { vec![a, b] }
            }
            _ => {
                let f = bz.expect("cubic entries come from the Bezier fit").fits[e.index];
                let f = if kind == 1 { f } else { [f[3], f[2], f[1], f[0]] };
                (0..=steps)
                    .map(|i| {
                        let t = i as f64 / steps as f64;
                        let u = 1.0 - t;
                        let b = [u * u * u, 3.0 * u * u * t, 3.0 * u * t * t, t * t * t];
                        [0, 1].map(|d| (0..4).map(|k| b[k] * f[k][d]).sum())
                    })
                    .collect()
            }
        };
        let skip = usize::from(out.last() == pts.first());
        out.extend_from_slice(&pts[skip..]);
    };
    let mut own = Vec::new();
    for &e in &ents[s] {
        push(&mut own, s, e, false);
    }
    let mut out = vec![own];
    for ring in &holes(g, &ents)[s] {
        let mut l = Vec::new();
        for &(c, j) in ring.iter().rev() {
            push(&mut l, c, ents[c][j], true);
        }
        out.push(l);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stages::boundary::trace_boundaries;

    fn square_with_hole() -> BoundaryGraph {
        let (w, h) = (8usize, 8usize);
        let mut l = vec![0u32; w * h];
        for y in 2..6 {
            for x in 2..6 {
                l[y * w + x] = 1;
            }
        }
        trace_boundaries(w, h, &l, 2, false)
    }

    #[test]
    fn hole_is_the_child_ring_reversed() {
        let g = square_with_hole();
        let colors = [[255, 255, 255, 255], [0, 0, 0, 255]];
        let svg = export_svg(&g, None, &colors, ExportOptions::default());
        assert!(svg.starts_with("<?xml"));
        assert!(svg.contains("<svg width=\"8pt\" height=\"8pt\" viewBox=\"0 0 8 8\""));
        // Background: outer border then one hole (two M commands in its path).
        let bg = svg.lines().find(|l| l.starts_with("<path fill=\"#ffffff\"")).expect("background path");
        assert_eq!(bg.matches(" M ").count(), 2, "{bg}");
        assert_eq!(svg.matches("<g id=").count(), 2);
        assert!(svg.ends_with("</g>\n</svg>\n"));
    }

    #[test]
    fn stacked_mode_has_no_holes_or_groups() {
        let g = square_with_hole();
        let colors = [[255, 255, 255, 255], [0, 0, 0, 255]];
        let svg = export_svg(&g, None, &colors, ExportOptions { mode: ShapeMode::Stacked, stroke: true });
        assert_eq!(svg.matches(" M ").count(), 2);
        assert!(!svg.contains("<g "));
        assert!(svg.contains("stroke-width=\"0.09375\""));
    }

    /// A hidden segment loses its path only: the parent keeps the hole, the group tags stay.
    #[test]
    fn hidden_segment_keeps_parent_hole() {
        let g = square_with_hole();
        let colors = [[255, 255, 255, 255], [0, 0, 0, 255]];
        let svg = export_svg_hidden(&g, None, &colors, ExportOptions::default(), &[false, true]);
        assert!(!svg.contains("<path fill=\"#000000\""));
        let bg = svg.lines().find(|l| l.starts_with("<path fill=\"#ffffff\"")).expect("background path");
        assert_eq!(bg.matches(" M ").count(), 2, "{bg}");
        assert_eq!(svg.matches("<g id=").count(), 2);
        assert_eq!(svg.matches("</g>").count(), 2);
        let all = export_svg_hidden(&g, None, &colors, ExportOptions::default(), &[false]);
        assert_eq!(all, export_svg(&g, None, &colors, ExportOptions::default()));
    }

    #[test]
    fn outline_has_own_loop_and_holes() {
        let g = square_with_hole();
        let o = outline(&g, None, 0, 8);
        assert_eq!(o.len(), 2);
        assert_eq!(outline(&g, None, 1, 8).len(), 1);
    }

    #[test]
    fn groups_follow_first_appearance() {
        let c = [[1, 1, 1, 255], [2, 2, 2, 255], [1, 1, 1, 255]];
        assert_eq!(colour_groups(&c), vec![(0, true, false), (2, false, true), (1, true, true)]);
    }
}
