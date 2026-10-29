//! Palette finder: proposes a palette of k colours for the fixed-palette modes.
//!
//! For every k in [from, to]:
//! a 4-D colour histogram (8 or 16 bins per channel) gives peaks
//! that seed a k-means; empty or missing clusters are seeded
//! from the engine LCG. Each palette is scored by `log10(Σ residual + 1) + (k − 2)·cost_per_color`
//! and its colours are rounded to bytes.
//!
//! For anti-aliased input pixels on edges weigh less in the histogram and the means
//! (`EdgeImage`), and pixels near a palette boundary are scored against the
//! best blend of the neighbouring palette colours.
//!
//! Everything runs in the engine's BGRA byte order; colours are converted at the API boundary.

use super::labels::{INV_255_F32, Lcg, NBR_DX, NBR_DY};
use super::subpixel::mix;
use crate::image::RgbaImage;
use crate::params::EngineConfig;

/// Tunables of the palette finder (always taken from the `pass1` config).
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct PaletteParams {
    /// Radius of the palette-boundary band scored with blends (default 2).
    pub boundary_radius: i32,
    pub cost_per_color: f32,
    pub use_16_hist_bins: bool,
    pub use_26_connected_peaks: bool,
    /// True in basic mode when `image_type == ArtworkBlended`.
    pub is_anti_aliased: bool,
    pub edge_weight_scale: f64,
    pub edge_weight_floor: f64,
}

impl PaletteParams {
    pub fn new(cfg: &EngineConfig, is_anti_aliased: bool) -> Self {
        Self {
            boundary_radius: 2,
            cost_per_color: cfg.cost_per_color,
            use_16_hist_bins: cfg.use_16_hist_bins != 0,
            use_26_connected_peaks: cfg.use_26_connected_peaks != 0,
            is_anti_aliased,
            edge_weight_scale: cfg.edge_weight_scale,
            edge_weight_floor: cfg.edge_weight_floor,
        }
    }
}

/// One proposed palette.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct FoundPalette {
    /// RGBA. Colours of clusters left empty by the last k-means update are (0, 0, 0, 0).
    pub colors: Vec<[u8; 4]>,
    /// log10(Σ squared residual + 1).
    pub fit_cost: f32,
    /// (k − 2)·cost_per_color.
    pub color_cost: f32,
    /// fit_cost + color_cost.
    pub cost: f32,
    /// Pixels per colour from the last assignment: 1 per pixel, or
    /// for anti-aliased input the pixel's edge strength / 255 (not its weight).
    pub counts: Vec<f64>,
}

/// Palettes for k = from..=to (to is raised to from if smaller).
pub fn find_palettes(img: &RgbaImage, p: &PaletteParams, from: usize, to: usize) -> Vec<FoundPalette> {
    find_palettes_with_peaks(img, p, from, to).0
}

/// [`find_palettes`] plus the histogram peak counts, most populated first.
pub fn find_palettes_with_peaks(
    img: &RgbaImage,
    p: &PaletteParams,
    from: usize,
    to: usize,
) -> (Vec<FoundPalette>, Vec<f64>) {
    let mut finder = PaletteFinder::new(img, p);
    let mut rng = Lcg(1);
    let pals = (from..=to.max(from)).map(|k| finder.run(k, &mut rng)).collect();
    (pals, finder.peak_counts)
}

/// Index of the cheapest palette (first minimum). After a 2..=12 run, the palette search
/// recommends it when the colour-count tree chose the flexible mode ("13"); otherwise it
/// recommends the tree's colour count (`classify::recommend_from_palettes`).
pub fn cheapest(palettes: &[FoundPalette]) -> Option<usize> {
    let mut best: Option<usize> = None;
    for (i, pal) in palettes.iter().enumerate() {
        if best.is_none_or(|b| pal.cost < palettes[b].cost) {
            best = Some(i);
        }
    }
    best
}

struct PaletteFinder<'a> {
    p: &'a PaletteParams,
    w: usize,
    h: usize,
    /// Working image, BGRA.
    px: Vec<[u8; 4]>,
    /// `EdgeImage` map (anti-aliased input only).
    edges: Option<Vec<u8>>,
    /// Histogram peaks, most populated first, and their counts.
    peaks: Vec<[u8; 4]>,
    peak_counts: Vec<f64>,
    /// Palette index per pixel.
    labels: Vec<usize>,
}

impl<'a> PaletteFinder<'a> {
    fn new(img: &RgbaImage, p: &'a PaletteParams) -> Self {
        let px: Vec<[u8; 4]> = img.pixels.iter().map(|c| [c.b, c.g, c.r, c.a]).collect();
        let (w, h) = (img.width, img.height);
        let edges = p.is_anti_aliased.then(|| edge_image(&px, w, h));
        let mut f = Self { p, w, h, px, edges, peaks: Vec::new(), peak_counts: Vec::new(), labels: vec![0; w * h] };
        (f.peaks, f.peak_counts) = f.histogram_peaks().into_iter().unzip();
        f
    }

    fn weight(&self, i: usize) -> f64 {
        let e = self.edges.as_ref().map_or(0, |m| m[i]);
        edge_weight(e, self.p.edge_weight_scale, self.p.edge_weight_floor)
    }

    /// Weighted histogram and its peaks, most populated first.
    fn histogram_peaks(&self) -> Vec<([u8; 4], f64)> {
        let mut hist = Histogram::new(if self.p.use_16_hist_bins { 16 } else { 8 });
        for (i, &c) in self.px.iter().enumerate() {
            hist.add(c, if self.edges.is_some() { self.weight(i) } else { 1.0 });
        }
        hist.peaks(self.p.use_26_connected_peaks)
    }

    fn pixel(&self, i: usize) -> [f32; 4] {
        self.px[i].map(|v| f32::from(v) * INV_255_F32)
    }

    /// K-means palette of k colours, its cost, then rounding to bytes.
    fn run(&mut self, k: usize, rng: &mut Lcg) -> FoundPalette {
        let mut pal = self.init(k, rng);
        let mut converged = false;
        let mut weights = vec![0.0f32; k];
        let mut counts = vec![0.0f64; k];
        for it in 0..10 {
            self.assign(&pal);
            if it == 9 || converged {
                counts = self.counts(k);
            }
            if converged {
                break;
            }
            let (centres, w) = self.update(k, rng);
            let mut moved = 0.0f32;
            for (old, new) in pal.iter().zip(&centres) {
                let d = (0..4).map(|i| (old[i] - new[i]).abs()).fold(0.0f32, f32::max);
                moved = moved.max(d);
            }
            pal = centres;
            weights = w;
            converged = moved < 0.01;
        }
        for (c, &w) in pal.iter_mut().zip(&weights) {
            if w == 0.0 {
                *c = [0.0; 4];
            }
        }
        let (fit_cost, color_cost, cost) = self.cost(&pal);
        let colors = pal
            .iter()
            .map(|c| {
                let b = c.map(|v| (f64::from(v) * 255.0 + 0.5) as i32 as u8);
                [b[2], b[1], b[0], b[3]]
            })
            .collect();
        FoundPalette { colors, fit_cost, color_cost, cost, counts }
    }

    /// Number of pixels assigned to each of the k clusters.
    fn counts(&self, k: usize) -> Vec<f64> {
        let mut c = vec![0.0f64; k];
        for (i, &l) in self.labels.iter().enumerate() {
            c[l] += match &self.edges {
                Some(e) => f64::from(e[i]) * 0.00392156862745098,
                None => 1.0,
            };
        }
        c
    }

    /// The first k peaks, then random opaque colours.
    fn init(&self, k: usize, rng: &mut Lcg) -> Vec<[f32; 4]> {
        (0..k)
            .map(|i| match self.peaks.get(i) {
                Some(c) => c.map(|v| f32::from(v) * INV_255_F32),
                None => random_color(rng),
            })
            .collect()
    }

    fn assign(&mut self, pal: &[[f32; 4]]) {
        assign_nearest(&self.px, pal, &mut self.labels);
    }

    /// Cluster means (plain or edge-weighted); an empty cluster
    /// gets a random opaque colour.
    fn update(&self, k: usize, rng: &mut Lcg) -> (Vec<[f32; 4]>, Vec<f32>) {
        let mut sum = vec![[0.0f32; 4]; k];
        let mut wsum = vec![0.0f32; k];
        for (i, &l) in self.labels.iter().enumerate() {
            let c = self.pixel(i);
            let w = if self.edges.is_some() { self.weight(i) as f32 } else { 1.0 };
            for j in 0..4 {
                sum[l][j] += if self.edges.is_some() { c[j] * w } else { c[j] };
            }
            wsum[l] += w;
        }
        let centres = sum
            .iter()
            .zip(&wsum)
            .map(|(s, &n)| {
                if n <= 0.0 {
                    random_color(rng)
                } else {
                    let inv = 1.0 / n;
                    s.map(|v| inv * v)
                }
            })
            .collect();
        (centres, wsum)
    }

    /// Returns (log10(Σ residual + 1), (k − 2)·cost_per_color, total).
    fn cost(&self, pal: &[[f32; 4]]) -> (f32, f32, f32) {
        let band = boundary_band(&self.labels, self.w, self.h, self.p.boundary_radius);
        let mut sum = 0.0f32;
        for y in 0..self.h {
            for x in 0..self.w {
                sum += self.residual(pal, &band, x, y);
            }
        }
        let fit = ((f64::from(sum) + 1.0).log2() * 0.3010299956639812) as f32;
        let colors = (pal.len() as i32 - 2) as f32 * self.p.cost_per_color;
        (fit, colors, colors + fit)
    }

    /// Squared RGBA error of the pixel against its palette colour, or (anti-aliased
    /// input, inside the boundary band) against the best blend of the 8-neighbourhood's colours.
    fn residual(&self, pal: &[[f32; 4]], band: &[i8], x: usize, y: usize) -> f32 {
        let i = y * self.w + x;
        let c = self.pixel(i);
        let model = if self.edges.is_none() || band[i] < 1 {
            pal[self.labels[i]]
        } else {
            // Candidates: the pixel's own palette colour and those of its neighbours.
            let mut cands = vec![self.labels[i]];
            for d in 0..8 {
                let (nx, ny) = (x as i32 + NBR_DX[d], y as i32 + NBR_DY[d]);
                if nx < 0 || ny < 0 || nx >= self.w as i32 || ny >= self.h as i32 {
                    continue;
                }
                let l = self.labels[ny as usize * self.w + nx as usize];
                if !cands.contains(&l) {
                    cands.push(l);
                    if cands.len() > 8 {
                        break;
                    }
                }
            }
            let cols: Vec<[f32; 4]> = cands.iter().map(|&l| pal[l]).collect();
            mix(&cols, c).1
        };
        let d = [0, 1, 2, 3].map(|j| model[j] - c[j]);
        d[0] * d[0] + d[1] * d[1] + d[2] * d[2] + d[3] * d[3]
    }
}

/// Nearest palette colour per pixel (BGRA, squared distance in f32, first wins
/// ties).
fn assign_nearest(px: &[[u8; 4]], pal: &[[f32; 4]], out: &mut [usize]) {
    for (i, p) in px.iter().enumerate() {
        let c = p.map(|v| f32::from(v) * INV_255_F32);
        let dist = |q: &[f32; 4]| {
            let d = [0, 1, 2, 3].map(|j| c[j] - q[j]);
            d[0] * d[0] + d[1] * d[1] + d[2] * d[2] + d[3] * d[3]
        };
        let mut best = 0;
        let mut best_d = dist(&pal[0]);
        for (j, q) in pal.iter().enumerate().skip(1) {
            let d = dist(q);
            if d < best_d {
                best = j;
                best_d = d;
            }
        }
        out[i] = best;
    }
}

/// Index of the nearest colour of `palette` (RGBA, non-empty) for every pixel.
pub(crate) fn nearest_palette_labels(img: &RgbaImage, palette: &[[u8; 4]]) -> Vec<usize> {
    let px: Vec<[u8; 4]> = img.pixels.iter().map(|c| [c.b, c.g, c.r, c.a]).collect();
    let pal: Vec<[f32; 4]> =
        palette.iter().map(|c| [c[2], c[1], c[0], c[3]].map(|v| f32::from(v) * INV_255_F32)).collect();
    let mut out = vec![0; px.len()];
    assign_nearest(&px, &pal, &mut out);
    out
}

/// `max(floor, 1 − e·scale/255)`.
pub(crate) fn edge_weight(e: u8, scale: f64, floor: f64) -> f64 {
    let v = 1.0 - f64::from(e) * scale * 0.00392156862745098;
    if v < floor { floor } else { v }
}

/// `ColorHistogram`: 4-D histogram of BGRA bytes with 8 or 16 bins per channel (f64 counts).
#[derive(Clone)]
pub(crate) struct Histogram {
    nb: usize,
    shift: u32,
    data: Vec<f64>,
}

impl Histogram {
    /// 8 bins (shift 5) or 16 bins (shift 4).
    pub(crate) fn new(nb: usize) -> Self {
        Self { nb, shift: if nb == 8 { 5 } else { 4 }, data: vec![0.0; nb * nb * nb * nb] }
    }

    fn index(&self, c: [usize; 4]) -> usize {
        ((c[3] * self.nb + c[2]) * self.nb + c[1]) * self.nb + c[0]
    }

    fn bin(&self, c: [u8; 4]) -> [usize; 4] {
        c.map(|v| usize::from(v >> self.shift))
    }

    /// Add weight `w` for colour `c`.
    pub(crate) fn add(&mut self, c: [u8; 4], w: f64) {
        let k = self.index(self.bin(c));
        self.data[k] += w;
    }

    /// Sum of all bins in order.
    pub(crate) fn total(&self) -> f64 {
        self.data.iter().fold(0.0, |acc, &v| acc + v)
    }

    /// Clear the bin of `c`, or with `wide` its whole 3⁴ neighbourhood.
    pub(crate) fn zero(&mut self, c: [u8; 4], wide: bool) {
        let b = self.bin(c);
        if !wide {
            let k = self.index(b);
            self.data[k] = 0.0;
            return;
        }
        let n = self.nb as i64;
        for d0 in -1..=1i64 {
            for d1 in -1..=1i64 {
                for d2 in -1..=1i64 {
                    for d3 in -1..=1i64 {
                        let q = [b[0] as i64 + d0, b[1] as i64 + d1, b[2] as i64 + d2, b[3] as i64 + d3];
                        if q.iter().all(|&t| (0..n).contains(&t)) {
                            let k = self.index(q.map(|t| t as usize));
                            self.data[k] = 0.0;
                        }
                    }
                }
            }
        }
    }

    /// A peak is a bin holding ≥ 2 whose neighbours (L1 distance 1, or all 80 in
    /// 26-connected mode) are not larger. Peaks are reported at the bin centre, sorted by count.
    pub(crate) fn peaks(&self, all: bool) -> Vec<([u8; 4], f64)> {
        let n = self.nb as i32;
        let shift = self.shift;
        let mut peaks: Vec<([u8; 4], f64)> = Vec::new();
        for c3 in 0..n {
            for c2 in 0..n {
                for c1 in 0..n {
                    for c0 in 0..n {
                        let v = self.data[self.index([c0, c1, c2, c3].map(|c| c as usize))];
                        if v < 2.0 {
                            continue;
                        }
                        let mut is_peak = true;
                        'nbr: for d3 in -1..=1i32 {
                            for d2 in -1..=1i32 {
                                for d1 in -1..=1i32 {
                                    for d0 in -1..=1i32 {
                                        if !all && d0.abs() + d1.abs() + d2.abs() + d3.abs() != 1 {
                                            continue;
                                        }
                                        let q = [c0 + d0, c1 + d1, c2 + d2, c3 + d3];
                                        if q.iter().any(|&t| t < 0 || t >= n) {
                                            continue;
                                        }
                                        if v < self.data[self.index(q.map(|t| t as usize))] {
                                            is_peak = false;
                                            break 'nbr;
                                        }
                                    }
                                }
                            }
                        }
                        if is_peak {
                            let half = 1u8 << (shift - 1);
                            peaks.push(([c0, c1, c2, c3].map(|c| ((c as u8) << shift).wrapping_add(half)), v));
                        }
                    }
                }
            }
        }
        // Merge sort with this predicate: higher count first;
        // on a tie `a` goes first if any byte of `b` is smaller (not a strict order).
        list_merge_sort(peaks, |a, b| a.1 > b.1 || (a.1 == b.1 && (0..4).any(|i| b.0[i] < a.0[i])))
    }
}

/// Random opaque colour drawn from the LCG: channel 2 first, then 1, then 0.
fn random_color(rng: &mut Lcg) -> [f32; 4] {
    let c2 = rng.next() as f32;
    let c1 = rng.next() as f32;
    let c0 = rng.next() as f32;
    [c0, c1, c2, 1.0]
}

/// Edge strength per pixel, `trunc(clamp(√S·255/√12, 0, 255))` where S sums the
/// squared channel differences (×1/255) to the pixels above, below, left and right (mirrored at the
/// border).
pub(crate) fn edge_image(px: &[[u8; 4]], w: usize, h: usize) -> Vec<u8> {
    let f = |c: [u8; 4]| c.map(|v| f32::from(v) * INV_255_F32);
    let mut out = vec![0u8; w * h];
    for y in 0..h as i32 {
        let up = if y <= 0 { y + 1 } else { y - 1 };
        let down = if y < h as i32 - 1 { y + 1 } else { y - 1 };
        for x in 0..w as i32 {
            let left = if x <= 0 { x + 1 } else { x - 1 };
            let right = if x < w as i32 - 1 { x + 1 } else { x - 1 };
            let at = |xx: i32, yy: i32| f(px[yy as usize * w + xx as usize]);
            let c = at(x, y);
            let mut s = 0.0f64;
            for n in [at(x, up), at(x, down), at(left, y), at(right, y)] {
                for ch in 0..4 {
                    let d = f64::from(n[ch] - c[ch]);
                    s += d * d;
                }
            }
            let v = (s.sqrt() * 255.0 / 12.0f64.sqrt()).clamp(0.0, 255.0);
            out[y as usize * w + x as usize] = v as u8;
        }
    }
    out
}

/// `r` on both sides of every label change (2×2 blocks), then rings r−1 … 1.
pub(crate) fn boundary_band(labels: &[usize], w: usize, h: usize, r: i32) -> Vec<i8> {
    let mut m = vec![0i8; w * h];
    let blocks = |y: usize, x: usize| [y * w + x, y * w + x + 1, (y + 1) * w + x, (y + 1) * w + x + 1];
    for y in 0..h.saturating_sub(1) {
        for x in 0..w.saturating_sub(1) {
            let l = labels[y * w + x];
            if l != labels[y * w + x + 1] || l != labels[(y + 1) * w + x] || l != labels[(y + 1) * w + x + 1] {
                for q in blocks(y, x) {
                    m[q] = r as i8;
                }
            }
        }
    }
    let mut v = r - 1;
    while v > 0 {
        for y in 0..h.saturating_sub(1) {
            for x in 0..w.saturating_sub(1) {
                let b = blocks(y, x);
                if b.iter().any(|&q| i32::from(m[q]) > v) {
                    for q in b {
                        if m[q] == 0 {
                            m[q] = v as i8;
                        }
                    }
                }
            }
        }
        v -= 1;
    }
    m
}

/// A bin merge sort, kept exactly because the peak predicate is
/// not a strict weak order and the result depends on the merge sequence.
fn list_merge_sort<T>(items: Vec<T>, pred: impl Fn(&T, &T) -> bool) -> Vec<T> {
    const MAX_BINS: usize = 25;
    if items.len() < 2 {
        return items;
    }
    // `list::merge`: elements of `right` go before the first element of `left` they beat.
    let merge = |left: Vec<T>, right: Vec<T>| -> Vec<T> {
        let mut out = Vec::with_capacity(left.len() + right.len());
        let mut r = right.into_iter().peekable();
        for l in left {
            while r.peek().is_some_and(|x| pred(x, &l)) {
                out.push(r.next().unwrap());
            }
            out.push(l);
        }
        out.extend(r);
        out
    };
    let mut bins: Vec<Vec<T>> = (0..=MAX_BINS).map(|_| Vec::new()).collect();
    let mut max_bin = 0;
    for item in items {
        let mut carry = vec![item];
        let mut bin = 0;
        while bin < max_bin && !bins[bin].is_empty() {
            carry = merge(std::mem::take(&mut bins[bin]), carry);
            bin += 1;
        }
        if bin == MAX_BINS {
            bins[bin - 1] = merge(std::mem::take(&mut bins[bin - 1]), carry);
        } else {
            bins[bin] = carry;
            if bin == max_bin {
                max_bin += 1;
            }
        }
    }
    for bin in 1..max_bin {
        let lower = std::mem::take(&mut bins[bin - 1]);
        bins[bin] = merge(std::mem::take(&mut bins[bin]), lower);
    }
    std::mem::take(&mut bins[max_bin - 1])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::image::Rgba;

    fn params(aa: bool) -> PaletteParams {
        PaletteParams::new(&EngineConfig::pass1(), aa)
    }

    fn two_colour_image() -> RgbaImage {
        let (w, h) = (12, 8);
        let pixels =
            (0..w * h)
                .map(|i| {
                    if i % w < 5 {
                        Rgba { r: 200, g: 30, b: 30, a: 255 }
                    } else {
                        Rgba { r: 20, g: 40, b: 220, a: 255 }
                    }
                })
                .collect();
        RgbaImage::new(w, h, pixels)
    }

    #[test]
    fn list_sort_is_stable_for_a_strict_order() {
        let v: Vec<(i32, usize)> = [3, 1, 3, 2, 1, 3, 0, 2].iter().copied().zip(0..).collect();
        let s = list_merge_sort(v, |a, b| a.0 > b.0);
        assert_eq!(s.iter().map(|x| x.1).collect::<Vec<_>>(), vec![0, 2, 5, 3, 7, 1, 4, 6]);
    }

    #[test]
    fn two_flat_colours_are_found_exactly() {
        let pals = find_palettes(&two_colour_image(), &params(false), 2, 3);
        let mut c = pals[0].colors.clone();
        c.sort();
        assert_eq!(c, vec![[20, 40, 220, 255], [200, 30, 30, 255]]);
        assert!(pals[0].fit_cost < 1e-9); // f32 means: Σ·(1/n) is not exact
        // k = 3 has only two peaks: the third colour is random and ends up unused (zeroed).
        assert!(pals[1].colors.contains(&[0, 0, 0, 0]));
        assert!(pals[1].cost > pals[0].cost);
        assert_eq!(cheapest(&pals), Some(0));
    }

    #[test]
    fn anti_aliased_mode_scores_the_boundary_with_blends() {
        let mut img = two_colour_image();
        // A one-pixel blended column between the two colours.
        for y in 0..img.height {
            img.pixels[y * img.width + 5] = Rgba { r: 110, g: 35, b: 125, a: 255 };
        }
        let plain = find_palettes(&img, &params(false), 2, 2);
        let aa = find_palettes(&img, &params(true), 2, 2);
        assert!(aa[0].fit_cost < plain[0].fit_cost);
    }

    #[test]
    fn edge_strength_of_a_step() {
        let px = vec![[0, 0, 0, 255], [255, 255, 255, 255], [0, 0, 0, 255], [255, 255, 255, 255]];
        // Each pixel sees the other column twice (mirrored) and its own colour vertically.
        let e = edge_image(&px, 2, 2);
        // S = 2·3 (RGB fully different) → √6·255/√12 = 180.3.
        assert_eq!(e, vec![180; 4]);
    }
}
