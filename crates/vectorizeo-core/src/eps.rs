//! Encapsulated PostScript writer. The shapes,
//! their order and their hole loops are exactly those of the SVG, in the same coordinates
//! (image pixels, y down) under a y-flipping transform.
//!
//! PostScript has no transparency, so a shape's opacity is applied over white (what the SVG
//! shows on a white page); shapes the SVG writer leaves out (alpha ≤ 10) are left out here too.
//! Paths are filled with the nonzero rule like SVG's default (hole loops run the other way).

use std::fmt::Write;

use crate::export::VectorWriter;
use crate::svg::{P, fixed2};

/// The SVG writer's hairline stroke width, in image units.
const STROKE_WIDTH: &str = "0.09375";

pub struct EpsWriter {
    out: String,
    stroke: bool,
    /// A loop of the current path is open (closed before the next one starts).
    open_loop: bool,
}

/// Channel value in 0..1 with the opacity applied over white.
fn over_white(v: u8, opacity: f64) -> f64 {
    f64::from(v) / 255.0 * opacity + (1.0 - opacity)
}

impl EpsWriter {
    /// `view`: image size in units (the SVG viewBox); `size_pt`: page size in points.
    pub fn header(view: [i32; 2], size_pt: [i32; 2], stroke: bool) -> Self {
        let mut out = String::new();
        let scale = |d: usize| if view[d] > 0 { f64::from(size_pt[d]) / f64::from(view[d]) } else { 1.0 };
        out.push_str("%!PS-Adobe-3.0 EPSF-3.0\n");
        out.push_str("%%Creator: vectorizeo\n");
        let _ = writeln!(out, "%%BoundingBox: 0 0 {} {}", size_pt[0], size_pt[1]);
        let _ = writeln!(out, "%%HiResBoundingBox: 0 0 {} {}", size_pt[0], size_pt[1]);
        out.push_str("%%LanguageLevel: 2\n%%Pages: 1\n%%EndComments\n");
        out.push_str("%%BeginProlog\n");
        out.push_str("/vectorizeo 8 dict def vectorizeo begin\n");
        out.push_str("/m /moveto load def /l /lineto load def /c /curveto load def /h /closepath load def\n");
        out.push_str("/rg /setrgbcolor load def /f /fill load def\n");
        out.push_str("/fs { gsave fill grestore stroke } bind def\n");
        out.push_str("end\n%%EndProlog\n");
        out.push_str("%%Page: 1 1\n");
        out.push_str("vectorizeo begin\ngsave\n");
        let _ = writeln!(out, "0 {} translate {} {} scale", size_pt[1], scale(0), -scale(1));
        if stroke {
            let _ = writeln!(out, "{STROKE_WIDTH} setlinewidth 1 setlinejoin");
        }
        Self { out, stroke, open_loop: false }
    }

    fn point(&mut self, p: P) {
        let _ = write!(self.out, "{} {}", fixed2(p[0]), fixed2(p[1]));
    }

    fn start_loop(&mut self, p: P) {
        if self.open_loop {
            self.out.push_str("h\n");
        }
        self.point(p);
        self.out.push_str(" m\n");
        self.open_loop = true;
    }
}

impl VectorWriter for EpsWriter {
    fn group_open(&mut self, c: [u8; 4]) {
        let _ = writeln!(self.out, "% group #{:02x}{:02x}{:02x}{:02x}", c[0], c[1], c[2], c[3]);
    }

    fn group_close(&mut self) {
        self.out.push_str("% end group\n");
    }

    fn path_open(&mut self, c: [u8; 4]) {
        // Same opacity steps as the SVG writer: 0xfb and above count as opaque.
        let opacity = if c[3] < 0xfb { f64::from(c[3]) / 255.0 } else { 1.0 };
        let [r, g, b] = [c[0], c[1], c[2]].map(|v| over_white(v, opacity));
        let _ = writeln!(self.out, "{r:.4} {g:.4} {b:.4} rg");
        self.out.push_str("newpath\n");
        self.open_loop = false;
    }

    fn path_close(&mut self) {
        self.out.push_str(if self.stroke { "h fs\n" } else { "h f\n" });
        self.open_loop = false;
    }

    fn line(&mut self, a: P, b: P, first: bool) {
        if first {
            self.start_loop(a);
        }
        self.point(b);
        self.out.push_str(" l\n");
    }

    fn cubic(&mut self, c: &[P; 4], first: bool) {
        if first {
            self.start_loop(c[0]);
        }
        for p in &c[1..] {
            self.point(*p);
            self.out.push(' ');
        }
        self.out.push_str("c\n");
    }

    fn finish(mut self) -> String {
        self.out.push_str("grestore\nend\nshowpage\n%%EOF\n");
        self.out
    }
}

#[cfg(test)]
mod tests {
    use crate::export::{ExportOptions, Format, export};
    use crate::stages::boundary::trace_boundaries;

    #[test]
    fn eps_has_one_filled_path_per_shape() {
        let (w, h) = (8usize, 8usize);
        let mut l = vec![0u32; w * h];
        for y in 2..6 {
            for x in 2..6 {
                l[y * w + x] = 1;
            }
        }
        let g = trace_boundaries(w, h, &l, 2, false);
        let colors = [[255, 255, 255, 255], [0, 0, 0, 128]];
        let eps = export(&g, None, &colors, ExportOptions::default(), &[], Format::Eps);
        assert!(eps.starts_with("%!PS-Adobe-3.0 EPSF-3.0\n"));
        assert!(eps.contains("%%BoundingBox: 0 0 8 8\n"));
        assert!(eps.contains("0 8 translate 1 -1 scale\n"));
        assert_eq!(eps.matches("newpath").count(), 2);
        assert_eq!(eps.matches("h f\n").count(), 2);
        // Background: outer loop plus one hole loop.
        let bg = eps.split("newpath").nth(1).unwrap();
        assert_eq!(bg.matches(" m\n").count(), 2);
        // Half-transparent black over white.
        assert!(eps.contains("0.4980 0.4980 0.4980 rg"), "{eps}");
        assert!(eps.ends_with("showpage\n%%EOF\n"));
        // Hidden shapes are left out.
        let eps = export(&g, None, &colors, ExportOptions::default(), &[false, true], Format::Eps);
        assert_eq!(eps.matches("newpath").count(), 1);
    }

    #[test]
    fn format_from_extension() {
        assert_eq!(Format::from_extension("EPS"), Format::Eps);
        assert_eq!(Format::from_extension("svg"), Format::Svg);
    }
}
