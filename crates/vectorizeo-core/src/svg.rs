//! `SvgWriter`: one method per kind of element. Coordinates are written with a fixed
//! precision of 2 digits, i.e. C `%.2f`.

use std::fmt::Write;

pub type P = [f64; 2];

/// C `%.2f`: the formatter converts to 17 significant digits and rounds that digit string half
/// up at the requested position (Rust's own formatting rounds the exact binary value instead).
pub fn fixed2(v: f64) -> String {
    let neg = v.is_sign_negative();
    let a = v.abs();
    if a == 0.0 {
        return if neg { "-0.00".into() } else { "0.00".into() };
    }
    // d.dddddddddddddddde±x
    let sci = format!("{a:.16e}");
    let (mant, exp) = sci.split_once('e').expect("exponent");
    let exp: i32 = exp.parse().expect("exponent value");
    let digits: Vec<u8> = mant.bytes().filter(u8::is_ascii_digit).map(|b| b - b'0').collect();
    // Digits kept: everything left of the point plus two decimals.
    let keep = exp + 3;
    let mut out: Vec<u8> = if keep <= 0 {
        // Rounds to 0.00 or 0.01.
        let up = keep == 0 && digits[0] >= 5;
        vec![0, 0, u8::from(up)]
    } else {
        let keep = keep as usize;
        let mut d: Vec<u8> = (0..keep).map(|i| digits.get(i).copied().unwrap_or(0)).collect();
        if digits.get(keep).is_some_and(|&r| r >= 5) {
            let mut i = keep;
            loop {
                if i == 0 {
                    d.insert(0, 1);
                    break;
                }
                i -= 1;
                if d[i] == 9 {
                    d[i] = 0;
                } else {
                    d[i] += 1;
                    break;
                }
            }
        }
        while d.len() < 3 {
            d.insert(0, 0);
        }
        d
    };
    let split = out.len() - 2;
    let frac = out.split_off(split);
    let mut s = String::new();
    if neg {
        s.push('-');
    }
    s.extend(out.iter().map(|&d| char::from(b'0' + d)));
    s.push('.');
    s.extend(frac.iter().map(|&d| char::from(b'0' + d)));
    s
}

/// `#rrggbb` body (the engine stores BGRA).
fn rgb(c: [u8; 4]) -> String {
    format!("{:02x}{:02x}{:02x}", c[0], c[1], c[2])
}

pub struct SvgWriter {
    out: String,
    /// Whether to add a hairline stroke in the fill colour.
    stroke: bool,
}

impl SvgWriter {
    /// XML prolog and the `<svg>` element (no newline after it; every
    /// element starts with one).
    pub fn header(view: [i32; 2], size_pt: [i32; 2], stroke: bool) -> Self {
        let mut out = String::new();
        out.push_str("<?xml version=\"1.0\" encoding=\"UTF-8\" ?>\n");
        out.push_str(
            "<!DOCTYPE svg PUBLIC \"-//W3C//DTD SVG 1.1//EN\" \"http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd\">\n",
        );
        let _ = write!(
            out,
            "<svg width=\"{}pt\" height=\"{}pt\" viewBox=\"0 0 {} {}\" version=\"1.1\" xmlns=\"http://www.w3.org/2000/svg\">",
            size_pt[0], size_pt[1], view[0], view[1]
        );
        Self { out, stroke }
    }

    /// Shapes with alpha ≤ 10 are not written.
    pub fn visible(c: [u8; 4]) -> bool {
        c[3] > 10
    }

    /// Open a colour group.
    pub fn group_open(&mut self, c: [u8; 4]) {
        let _ = write!(self.out, "\n<g id=\"#{}{:02x}\">", rgb(c), c[3]);
    }

    /// Close the current group.
    pub fn group_close(&mut self) {
        self.out.push_str("\n</g>");
    }

    /// Open a path in colour `c` (opacity from its alpha).
    pub fn path_open(&mut self, c: [u8; 4]) {
        let col = rgb(c);
        let _ = write!(self.out, "\n<path fill=\"#{col}");
        if self.stroke {
            let _ = write!(self.out, "\" stroke=\"#{col}\" stroke-width=\"0.09375");
        }
        let opacity = if c[3] < 0xfb { fixed2(f64::from(c[3]) * 0.00392156862745098) } else { "1.00".into() };
        let _ = write!(self.out, "\" opacity=\"{opacity}\" d=\"");
    }

    /// Close the current path.
    pub fn path_close(&mut self) {
        self.out.push_str(" Z\" />");
    }

    fn point(&mut self, cmd: &str, p: P) {
        let _ = write!(self.out, "{cmd}{} {}", fixed2(p[0]), fixed2(p[1]));
    }

    /// A line segment: `first` is true for an outer loop or 2 for a hole loop.
    pub fn line(&mut self, a: P, b: P, first: bool) {
        if first {
            self.point(" M ", a);
        }
        self.point(" L ", b);
    }

    /// A cubic Bézier segment.
    pub fn cubic(&mut self, c: &[P; 4], first: bool) {
        if first {
            self.point(" M ", c[0]);
        }
        self.point(" C ", c[1]);
        self.point(" ", c[2]);
        self.point(" ", c[3]);
    }

    /// Close the document and return the text.
    pub fn finish(mut self) -> String {
        self.out.push_str("\n</svg>\n");
        self.out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixed2_rounds_half_up() {
        assert_eq!(fixed2(0.0), "0.00");
        assert_eq!(fixed2(12.0), "12.00");
        assert_eq!(fixed2(0.125), "0.13");
        // 17 digits: 2.6749999999999998 → no round-up.
        assert_eq!(fixed2(2.675), "2.67");
        assert_eq!(fixed2(99.996), "100.00");
        assert_eq!(fixed2(0.004), "0.00");
        assert_eq!(fixed2(0.005), "0.01");
        assert_eq!(fixed2(-0.001), "-0.00");
        assert_eq!(fixed2(1.0 / 3.0), "0.33");
    }
}
