//! Input raster representation.

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Rgba {
    pub r: u8,
    pub g: u8,
    pub b: u8,
    pub a: u8,
}

#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct RgbaImage {
    pub width: usize,
    pub height: usize,
    pub pixels: Vec<Rgba>,
}

impl RgbaImage {
    pub fn new(width: usize, height: usize, pixels: Vec<Rgba>) -> Self {
        assert_eq!(pixels.len(), width * height, "pixel buffer size mismatch");
        Self { width, height, pixels }
    }

    #[inline]
    pub fn get(&self, x: usize, y: usize) -> Rgba {
        self.pixels[y * self.width + x]
    }
}

impl RgbaImage {
    /// Premultiply by alpha: colour channels become `trunc(c · (a/255))`,
    /// evaluated in double precision.
    pub fn premultiply(&mut self) {
        for p in &mut self.pixels {
            let s = f64::from(p.a) * (1.0 / 255.0);
            let m = |c: u8| (f64::from(c) * s) as i32 as u8;
            (p.r, p.g, p.b) = (m(p.r), m(p.g), m(p.b));
        }
    }

    /// Flatten onto a background colour (white or custom): composite the
    /// premultiplied pixels over `bg` (RGBA in 0..1), `out = c + bg·(1 − a)` in single precision.
    pub fn flatten(&self, bg: [f32; 4]) -> RgbaImage {
        // `trunc(v·255 + 0.5)`, keeping the low byte.
        let byte = |v: f32| (f64::from(v) * 255.0 + 0.5) as i32 as u8;
        let unit = |c: u8| f32::from(c) * (1.0f32 / 255.0);
        let bg = bg.map(|v| unit(byte(v)));
        let pixels = self
            .pixels
            .iter()
            .map(|p| {
                let a = unit(p.a);
                let t = 1.0 - a;
                Rgba {
                    r: byte(bg[0] * t + unit(p.r)),
                    g: byte(bg[1] * t + unit(p.g)),
                    b: byte(bg[2] * t + unit(p.b)),
                    a: byte(bg[3] * t + a),
                }
            })
            .collect();
        RgbaImage::new(self.width, self.height, pixels)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn premultiply_truncates() {
        let mut img =
            RgbaImage::new(2, 1, vec![Rgba { r: 255, g: 100, b: 3, a: 128 }, Rgba { r: 9, g: 9, b: 9, a: 255 }]);
        img.premultiply();
        assert_eq!(img.pixels[0], Rgba { r: 128, g: 50, b: 1, a: 128 });
        assert_eq!(img.pixels[1], Rgba { r: 9, g: 9, b: 9, a: 255 });
    }

    #[test]
    fn flatten_over_white() {
        let img = RgbaImage::new(
            3,
            1,
            vec![
                Rgba { r: 0, g: 0, b: 0, a: 0 },
                Rgba { r: 10, g: 200, b: 77, a: 255 },
                Rgba { r: 64, g: 0, b: 0, a: 128 },
            ],
        );
        let out = img.flatten([1.0; 4]);
        assert_eq!(out.pixels[0], Rgba { r: 255, g: 255, b: 255, a: 255 });
        assert_eq!(out.pixels[1], Rgba { r: 10, g: 200, b: 77, a: 255 });
        assert_eq!(out.pixels[2], Rgba { r: 191, g: 127, b: 127, a: 255 });
    }
}
