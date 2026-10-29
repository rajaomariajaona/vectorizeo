//! Pixel segmenter: greedy region merging with an annealed boundary penalty.
//!
//! Steps: quadtree initial over-segmentation, λ schedule, merge passes (merge/relabel decision
//! by merge cost, pixel moves), renumbering (flood fill + statistics rebuild) and small-segment
//! recolouring for unblended artwork.
//!
//! Precision: segment sums are f32, the quadtree runs in f64.

use super::labels;
use super::palette;
use super::subpixel::{SubPixelParams, SubPixelSegmenter};
use crate::image::RgbaImage;
use crate::params::EngineConfig;

/// 1/255 as f32, used when adding a pixel to a segment.
const INV_255_F32: f32 = 0.003921569;
/// 1/255 as f64, used by the quadtree.
const INV_255_F64: f64 = 0.00392156862745098;
/// Root tile size of the quadtree.
const QUADTREE_TILE: i32 = 16;
/// Segments below this size are recoloured.
const SMALL_SEGMENT: i32 = 10;

/// Segmenter tunables.
#[derive(Clone, Debug, PartialEq)]
pub struct SegmenterParams {
    pub use_over_seg_diag_extr: bool,
    pub max_iterations: i32,
    pub lambda_pre: f32,
    pub lambda_initial: f32,
    pub lambda_final: f32,
    pub min_num_pixels: i32,
    pub num_iter_btwn_renum: i32,
    pub rise_fraction: f32,
    pub size_penalty_weight: f32,
    pub size_penalty_power: f32,
    /// Not part of the text config. The recolouring's nearest-colour
    /// search starts at `-value`; the default is 10.0, so with a non-negative cost no small
    /// segment is ever recoloured (only palettization remains).
    pub recolor_field_9: f32,
}

impl SegmenterParams {
    pub fn from_config(c: &EngineConfig) -> Self {
        Self {
            use_over_seg_diag_extr: c.use_over_seg_diag_extr != 0,
            max_iterations: c.seg_max_iterations,
            lambda_pre: c.lambda_pre,
            lambda_initial: c.lambda_initial,
            lambda_final: c.lambda_final,
            min_num_pixels: c.min_num_pixels,
            num_iter_btwn_renum: c.num_iter_btwn_renum,
            rise_fraction: c.rise_fraction,
            size_penalty_weight: c.size_penalty_weight,
            size_penalty_power: c.size_penalty_power,
            recolor_field_9: 10.0,
        }
    }
}

/// Per-segment statistics.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct SegStats {
    pub count: i32,
    pub sum: [f32; 4],
}

impl SegStats {
    fn add(&mut self, px: [u8; 4]) {
        self.count += 1;
        for (s, &c) in self.sum.iter_mut().zip(&px) {
            *s += f32::from(c) * INV_255_F32;
        }
    }

    fn remove(&mut self, px: [u8; 4]) {
        self.count -= 1;
        for (s, &c) in self.sum.iter_mut().zip(&px) {
            *s -= f32::from(c) * INV_255_F32;
        }
    }

    fn absorb(&mut self, other: &mut SegStats) {
        self.count += other.count;
        for (s, o) in self.sum.iter_mut().zip(&other.sum) {
            *s += *o;
        }
        *other = SegStats::default();
    }

    /// Mean colour in 0..255 units: `sum * (255 / count)`.
    fn mean_255(&self) -> [f32; 4] {
        let scale = 255.0f32 / self.count as f32;
        self.sum.map(|s| s * scale)
    }

    /// Mean colour in 0..1 units: `sum * (1 / count)`.
    fn mean_unit(&self) -> [f32; 4] {
        let scale = 1.0f32 / self.count as f32;
        self.sum.map(|s| s * scale)
    }
}

fn sq_norm(v: [f32; 4]) -> f32 {
    v[0] * v[0] + v[1] * v[1] + v[2] * v[2] + v[3] * v[3]
}

fn pixel_dist2(px: [u8; 4], mean: [f32; 4]) -> f32 {
    let d = [0, 1, 2, 3].map(|i| f32::from(px[i]) - mean[i]);
    sq_norm(d)
}

/// The result of segmentation: a label per pixel plus per-segment statistics.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Segmentation {
    pub width: usize,
    pub height: usize,
    pub labels: Vec<u32>,
    pub segments: Vec<SegStats>,
    /// Per-segment palette colours; otherwise colours are the segment means.
    pub colors: Option<Vec<[u8; 4]>>,
}

impl Segmentation {
    /// The palette colour, else the mean, as bytes `trunc(255·c + 0.5)`.
    pub fn segment_color(&self, label: usize) -> [u8; 4] {
        if let Some(c) = &self.colors {
            return c[label];
        }
        let s = &self.segments[label];
        if s.count == 0 {
            return [0; 4];
        }
        s.mean_unit().map(round_u8)
    }
}

/// `trunc(255·c + 0.5)`, low byte kept.
fn round_u8(c: f32) -> u8 {
    (f64::from(c) * 255.0 + 0.5) as i32 as u8
}

/// Truncates toward zero; the result is stored as a byte.
fn trunc_u8(v: f32) -> u8 {
    (v as i32) as u8
}

/// Working state: the `Segmenter` object (`this` = PixelSegmentation+4).
pub struct Segmenter<'a> {
    p: SegmenterParams,
    img: &'a RgbaImage,
    w: i32,
    h: i32,
    /// Pixel labels (`PixelMap` +0). May point at non-root labels until `find` compresses them.
    labels: Vec<i32>,
    /// Union-find parent per label; -1 marks a root.
    parent: Vec<i32>,
    segs: Vec<SegStats>,
    /// Palette colour per label once the segmentation is palettised.
    colors: Option<Vec<[u8; 4]>>,
    lambda: f32,
    lambda_ratio: f32,
}

impl<'a> Segmenter<'a> {
    pub fn new(img: &'a RgbaImage, p: SegmenterParams) -> Self {
        let (w, h) = (img.width as i32, img.height as i32);
        // Ratio = 2^(log2(λf/λi) / (max_iter * rise_fraction)), stored as f32.
        let exponent = (f64::from(p.lambda_final) / f64::from(p.lambda_initial)).log2()
            / (f64::from(p.max_iterations) * f64::from(p.rise_fraction));
        let lambda_ratio = exponent.exp2() as f32;
        Self {
            lambda: p.lambda_initial,
            lambda_ratio,
            p,
            img,
            w,
            h,
            labels: vec![0; (w * h) as usize],
            parent: Vec::new(),
            segs: Vec::new(),
            colors: None,
        }
    }

    fn px(&self, i: i32) -> [u8; 4] {
        let c = self.img.pixels[i as usize];
        [c.r, c.g, c.b, c.a]
    }

    /// Root lookup with one-level path compression into both tables.
    fn find(&mut self, p: i32) -> i32 {
        labels::find(&mut self.labels, &mut self.parent, p as usize)
    }

    /// + stats merge: `parent[b] = a`, A absorbs B.
    fn merge(&mut self, a: i32, b: i32) {
        self.parent[b as usize] = a;
        let mut sb = std::mem::take(&mut self.segs[b as usize]);
        self.segs[a as usize].absorb(&mut sb);
    }

    /// Move pixel `p` from segment `from` to segment `to`.
    fn move_pixel(&mut self, p: i32, from: i32, to: i32) {
        let px = self.px(p);
        self.labels[p as usize] = to;
        self.segs[to as usize].add(px);
        self.segs[from as usize].remove(px);
    }

    /// Ward merge cost minus optional size penalty.
    fn merge_cost(&self, a: i32, b: i32) -> f32 {
        let (sa, sb) = (&self.segs[a as usize], &self.segs[b as usize]);
        let sq_a = sq_norm(sa.sum);
        let sq_b = sq_norm(sb.sum);
        let joint = [0, 1, 2, 3].map(|i| sb.sum[i] + sa.sum[i]);
        let sq_ab = sq_norm(joint);
        let (na, nb) = (sa.count, sb.count);
        let delta = sq_a / na as f32 + sq_b / nb as f32 - sq_ab / (na + nb) as f32;
        if self.p.size_penalty_weight > 0.0 {
            // pow() in f64, result rounded to f32 at the comparison.
            let pw = f64::from(self.p.size_penalty_power);
            let inv = |n: i32| 1.0 / f64::from(n).powf(pw);
            let penalty = (inv(nb) + inv(na) - inv(na + nb)) * f64::from(self.p.size_penalty_weight);
            return (f64::from(delta) - penalty) as f32;
        }
        delta
    }

    /// Returns 1 = merged, 2 = moved p into B, 3 = moved q into A, 0 = nothing.
    fn decide(&mut self, p: i32, q: i32, a: i32, b: i32) -> u8 {
        let cost = self.merge_cost(a, b);
        let (na, nb) = (self.segs[a as usize].count, self.segs[b as usize].count);
        let min = self.p.min_num_pixels;
        if self.lambda > cost || (self.lambda == self.p.lambda_final && (na < min || nb < min)) {
            self.merge(a, b);
            return 1;
        }
        let mean_a = self.segs[a as usize].mean_255();
        let mean_b = self.segs[b as usize].mean_255();
        let (pp, pq) = (self.px(p), self.px(q));
        if pixel_dist2(pp, mean_a) > pixel_dist2(pp, mean_b) {
            self.move_pixel(p, a, b);
            return 2;
        }
        if pixel_dist2(pq, mean_b) > pixel_dist2(pq, mean_a) {
            self.move_pixel(q, b, a);
            return 3;
        }
        0
    }

    /// One sweep. Mode 0 = forward 4-neighbours, 1 = all 8 neighbours,
    /// 2 = force-merge segments smaller than `min_num_pixels`.
    fn pass(&mut self, mode: u8) {
        const FWD4: [(i32, i32); 2] = [(1, 0), (0, 1)];
        const ALL8: [(i32, i32); 8] = [(1, 0), (0, 1), (-1, 0), (0, -1), (-1, 1), (1, -1), (-1, -1), (1, 1)];
        let nbrs: &[(i32, i32)] = if mode == 1 { &ALL8 } else { &FWD4 };
        let (w, h) = (self.w, self.h);
        for y in 0..h {
            for x in 0..w {
                let p = y * w + x;
                let mut a = self.find(p);
                for &(dx, dy) in nbrs {
                    let (nx, ny) = (x + dx, y + dy);
                    if nx >= w || ny >= h || nx < 0 || ny < 0 {
                        continue;
                    }
                    let q = p + dy * w + dx;
                    let b = self.find(q);
                    if a == b {
                        continue;
                    }
                    if mode == 2 {
                        let min = self.p.min_num_pixels;
                        if self.segs[a as usize].count < min || self.segs[b as usize].count < min {
                            self.merge(a, b);
                            a = self.find(p);
                        }
                    } else if self.decide(p, q, a, b) != 0 {
                        a = self.find(p);
                    }
                }
            }
        }
    }

    /// Variance-threshold quadtree over 16×16 tiles.
    fn quadtree_init(&mut self) {
        let threshold = f64::from(self.p.lambda_pre);
        let mut next_label = 0i32;
        for ty in (0..self.h).step_by(QUADTREE_TILE as usize) {
            for tx in (0..self.w).step_by(QUADTREE_TILE as usize) {
                self.quad(tx, ty, QUADTREE_TILE, threshold, &mut next_label, true);
            }
        }
        self.parent = vec![-1; next_label as usize];
        self.segs = vec![SegStats::default(); next_label as usize];
    }

    /// Returns (sum, sumsq, n, cost) of the block; assigns labels to the blocks that stay whole.
    fn quad(
        &mut self,
        x: i32,
        y: i32,
        size: i32,
        thr: f64,
        next: &mut i32,
        top: bool,
    ) -> ([f64; 4], [f64; 4], i32, f64) {
        if x < 0 || x >= self.w || y < 0 || y >= self.h {
            return ([0.0; 4], [0.0; 4], 0, 0.0);
        }
        if size == 1 {
            let px = self.px(y * self.w + x);
            let s = px.map(|c| f64::from(c) * INV_255_F64);
            return (s, s.map(|v| v * v), 1, 0.0);
        }
        let half = size / 2;
        let kids = [(x, y), (x + half, y), (x, y + half), (x + half, y + half)];
        let (mut sum, mut sumsq, mut n) = ([0.0f64; 4], [0.0f64; 4], 0);
        let mut costs = [0.0f64; 4];
        for (i, &(kx, ky)) in kids.iter().enumerate() {
            let (s, sq, kn, c) = self.quad(kx, ky, half, thr, next, false);
            for j in 0..4 {
                sum[j] += s[j];
                sumsq[j] += sq[j];
            }
            n += kn;
            costs[i] = c;
        }
        let cost = block_cost(&sum, &sumsq, n);
        if thr < cost {
            for (i, &(kx, ky)) in kids.iter().enumerate() {
                if costs[i] <= thr {
                    self.assign_block(kx, ky, half, next);
                }
            }
        } else if top {
            self.assign_block(x, y, size, next);
        }
        (sum, sumsq, n, cost)
    }

    /// Label every in-bounds pixel of the block, then bump the counter.
    fn assign_block(&mut self, x: i32, y: i32, size: i32, next: &mut i32) {
        for yy in y..y + size {
            for xx in x..x + size {
                if xx >= 0 && xx < self.w && yy >= 0 && yy < self.h {
                    self.labels[(yy * self.w + xx) as usize] = *next;
                }
            }
        }
        *next += 1;
    }

    /// Relabel connected components in raster order,
    /// then rebuild segment statistics by re-adding every pixel in raster order.
    fn renumber(&mut self) {
        let n = (self.w * self.h) as usize;
        let roots: Vec<i32> = (0..n as i32).map(|p| self.find(p)).collect();
        let (new, count) = labels::relabel_components(self.w, self.h, &roots);
        // carries each label's palette index over to its components.
        if let Some(old) = &self.colors {
            let mut c = vec![[0u8; 4]; count];
            for (p, &l) in new.iter().enumerate() {
                c[l as usize] = old[roots[p] as usize];
            }
            self.colors = Some(c);
        }
        self.labels = new;
        self.parent = vec![-1; count];
        self.segs = vec![SegStats::default(); count];
        for p in 0..n as i32 {
            let px = self.px(p);
            let l = self.labels[p as usize] as usize;
            self.segs[l].add(px);
        }
    }

    /// Run the segmenter. `extended` adds the 8-neighbour pass and the diagonal fix.
    pub fn run(&mut self, extended: bool) {
        self.quadtree_init();
        self.renumber();
        let saved_min = self.p.min_num_pixels;
        if extended {
            self.p.min_num_pixels = 0;
        }
        for it in 0..self.p.max_iterations {
            self.pass(0);
            if (it + 1) % self.p.num_iter_btwn_renum == 0 {
                self.renumber();
            }
            self.lambda *= self.lambda_ratio;
            if self.lambda > self.p.lambda_final {
                self.lambda = self.p.lambda_final;
            }
        }
        if extended {
            self.pass(1);
            labels::fix_diagonals(self.img, &mut self.labels, &mut self.parent);
            self.p.min_num_pixels = saved_min;
        }
        self.renumber();
        if self.p.min_num_pixels > 0 {
            self.pass(2);
            self.renumber();
        }
    }

    /// Give segments under 10 pixels the colour of the nearest large segment
    /// (cost `n_small * |μ_small - μ_big|²`), then merge labels sharing a colour into one
    /// palette entry. Returns the palette; labels become palette indices.
    pub fn recolor_small_and_palettize(&mut self) -> Vec<[u8; 4]> {
        let nseg = self.segs.len();
        let mut colors: Vec<[u8; 4]> = self.segs.iter().map(|s| s.mean_255().map(trunc_u8)).collect();
        // Big segments are collected in an ordered set keyed by (count, index).
        let mut big: Vec<(i32, usize)> =
            (0..nseg).filter(|&k| self.segs[k].count >= SMALL_SEGMENT).map(|k| (self.segs[k].count, k)).collect();
        big.sort();
        for k in 0..nseg {
            let sk = self.segs[k];
            if sk.count >= SMALL_SEGMENT {
                continue;
            }
            let mk = sk.mean_unit();
            let mut best = -f64::from(self.p.recolor_field_9);
            let mut best_idx = k;
            for &(_, j) in &big {
                let mj = self.segs[j].mean_unit();
                let d = [0, 1, 2, 3].map(|i| mk[i] - mj[i]);
                let cost = f64::from(sq_norm(d)) * f64::from(sk.count as f32);
                if best > cost {
                    best = cost;
                    best_idx = j;
                }
            }
            if best_idx != k {
                colors[k] = colors[best_idx];
            }
        }
        // Palette::add returns the index of an existing identical colour or appends it.
        let mut palette: Vec<[u8; 4]> = Vec::new();
        let seg_to_pal: Vec<usize> = colors
            .iter()
            .map(|c| match palette.iter().position(|p| p == c) {
                Some(i) => i,
                None => {
                    palette.push(*c);
                    palette.len() - 1
                }
            })
            .collect();
        for p in 0..self.labels.len() {
            let root = self.find(p as i32) as usize;
            self.labels[p] = seg_to_pal[root] as i32;
        }
        self.parent = vec![-1; palette.len()];
        self.colors = Some(palette.clone());
        palette
    }

    /// Photo mode: every segment gets a palette entry for its mean colour,
    /// `trunc(255·Σ·(1/n) + 0.5)`. Labels are unchanged.
    pub fn generate_palette(&mut self) {
        self.colors = Some(self.segs.iter().map(|s| s.mean_unit().map(round_u8)).collect());
    }

    /// If `min_num_pixels` > 0, force-merge small segments again (the merged
    /// segment keeps the survivor's palette colour) and renumber.
    pub fn enforce_min_size(&mut self) {
        if self.p.min_num_pixels > 0 {
            self.pass(2);
            self.renumber();
        }
    }

    pub fn into_segmentation(mut self) -> Segmentation {
        self.renumber();
        Segmentation {
            width: self.w as usize,
            height: self.h as usize,
            labels: self.labels.iter().map(|&l| l as u32).collect(),
            segments: self.segs,
            colors: self.colors,
        }
    }
}

/// `(Σsumsq − |Σsum|²/n) / n`.
fn block_cost(sum: &[f64; 4], sumsq: &[f64; 4], n: i32) -> f64 {
    if n == 0 {
        return 0.0;
    }
    let n = f64::from(n);
    let sq = sum[0] * sum[0] + sum[1] * sum[1] + sum[2] * sum[2] + sum[3] * sum[3];
    ((sumsq[0] + sumsq[1] + sumsq[2] + sumsq[3]) - sq / n) / n
}

/// Committing the Edit Segmentation page: every pixel of the
/// edited colour image becomes a palette entry (exact RGBA,
/// first-seen order in raster order), each 4-connected component of one colour becomes a
/// segment (statistics re-added from `img`) and keeps that palette colour.
pub fn segmentation_from_colors(img: &RgbaImage, colors: &[[u8; 4]]) -> Segmentation {
    let (w, h) = (img.width, img.height);
    assert_eq!(colors.len(), w * h, "one colour per pixel");
    let mut index: std::collections::HashMap<[u8; 4], i32> = std::collections::HashMap::new();
    let mut palette: Vec<[u8; 4]> = Vec::new();
    let roots: Vec<i32> = colors
        .iter()
        .map(|&c| {
            *index.entry(c).or_insert_with(|| {
                palette.push(c);
                palette.len() as i32 - 1
            })
        })
        .collect();
    let (labels, count) = labels::relabel_components(w as i32, h as i32, &roots);
    let mut segments = vec![SegStats::default(); count];
    let mut seg_colors = vec![[0u8; 4]; count];
    for (p, &l) in labels.iter().enumerate() {
        let c = img.pixels[p];
        segments[l as usize].add([c.r, c.g, c.b, c.a]);
        seg_colors[l as usize] = palette[roots[p] as usize];
    }
    Segmentation {
        width: w,
        height: h,
        labels: labels.into_iter().map(|l| l as u32).collect(),
        segments,
        colors: Some(seg_colors),
    }
}

/// Fixed-palette segmentation (`palette` is RGBA, non-empty).
///
/// Aliased input: every pixel takes the nearest palette colour, diagonal contacts are
/// fixed, and each 4-connected component becomes a segment in its palette colour.
/// Anti-aliased input runs the sub-pixel segmenter and snaps
/// its segments to the palette (`SubPixelSegmenter::finish_fixed`).
pub fn segment_pixels_fixed(img: &RgbaImage, cfg: &EngineConfig, palette: &[[u8; 4]]) -> Segmentation {
    assert!(!palette.is_empty(), "fixed palette must not be empty");
    if cfg.is_anti_aliased != 0 {
        let mut s = Segmenter::new(img, SegmenterParams::from_config(cfg));
        s.run(false);
        let labels = s.into_segmentation().labels.iter().map(|&l| l as i32).collect();
        let mut sub = SubPixelSegmenter::new(img, labels, SubPixelParams::from_config(cfg));
        sub.run();
        sub.finish_fixed(palette);
        return sub.into_segmentation();
    }
    let (w, h) = (img.width as i32, img.height as i32);
    let mut pal_labels: Vec<i32> = palette::nearest_palette_labels(img, palette).iter().map(|&l| l as i32).collect();
    let mut parent = vec![-1; palette.len()];
    labels::fix_diagonals(img, &mut pal_labels, &mut parent);
    let (comp, n) = labels::relabel_components(w, h, &pal_labels);
    let mut segments = vec![SegStats::default(); n];
    let mut colors = vec![[0u8; 4]; n];
    for (p, &c) in comp.iter().enumerate() {
        let px = img.pixels[p];
        segments[c as usize].add([px.r, px.g, px.b, px.a]);
        colors[c as usize] = palette[pal_labels[p] as usize];
    }
    Segmentation {
        width: img.width,
        height: img.height,
        labels: comp.iter().map(|&l| l as u32).collect(),
        segments,
        colors: Some(colors),
    }
}

/// Flexible-palette segmentation.
pub fn segment_pixels(img: &RgbaImage, cfg: &EngineConfig) -> Segmentation {
    let params = SegmenterParams::from_config(cfg);
    if cfg.is_anti_aliased != 0 {
        let mut s = Segmenter::new(img, params);
        s.run(false);
        let labels = s.into_segmentation().labels.iter().map(|&l| l as i32).collect();
        let mut sub = SubPixelSegmenter::new(img, labels, SubPixelParams::from_config(cfg));
        sub.run();
        sub.finish();
        return sub.into_segmentation();
    }
    let extended = params.use_over_seg_diag_extr;
    let mut s = Segmenter::new(img, params);
    s.run(extended);
    if extended {
        s.recolor_small_and_palettize();
        s.renumber();
    } else {
        s.generate_palette();
    }
    s.enforce_min_size();
    s.into_segmentation()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Rgba;
    use crate::params::{ImageType, Quality};

    fn image_from(w: usize, h: usize, f: impl Fn(usize, usize) -> Rgba) -> RgbaImage {
        let mut px = Vec::with_capacity(w * h);
        for y in 0..h {
            for x in 0..w {
                px.push(f(x, y));
            }
        }
        RgbaImage::new(w, h, px)
    }

    const RED: Rgba = Rgba { r: 220, g: 30, b: 30, a: 255 };
    const BLUE: Rgba = Rgba { r: 20, g: 40, b: 200, a: 255 };
    const WHITE: Rgba = Rgba { r: 255, g: 255, b: 255, a: 255 };

    /// Same-colour 4-connected components become segments; a diagonal touch splits them.
    #[test]
    fn colours_to_segments() {
        let img = image_from(3, 2, |_, _| WHITE);
        let (a, b) = ([1, 2, 3, 255], [9, 9, 9, 255]);
        let seg = segmentation_from_colors(&img, &[a, b, a, b, a, a]);
        assert_eq!(seg.labels, vec![0, 1, 2, 3, 2, 2]);
        assert_eq!(seg.segments.iter().map(|s| s.count).collect::<Vec<_>>(), vec![1, 1, 3, 1]);
        assert_eq!((0..4).map(|k| seg.segment_color(k)).collect::<Vec<_>>(), vec![a, b, a, b]);
    }

    #[test]
    fn block_cost_is_variance() {
        let sum = [1.0, 0.0, 0.0, 0.0];
        let sumsq = [1.0, 0.0, 0.0, 0.0];
        assert_eq!(block_cost(&sum, &sumsq, 2), 0.25);
    }

    #[test]
    fn two_flat_regions_give_two_segments() {
        let img = image_from(40, 24, |x, _| if x < 20 { RED } else { BLUE });
        for (t, q) in [(ImageType::Photo, Quality::Medium), (ImageType::ArtworkUnblended, Quality::Medium)] {
            let seg = segment_pixels(&img, &EngineConfig::preset(t, q));
            assert_eq!(seg.segments.len(), 2, "{t:?}");
            // Unblended palettes truncate f32 `Σ·(255/n)`, which lands just below the integer
            // (e.g. 219 for a flat 220 region); photo palettes round `Σ·(1/n)·255`.
            let close = |a: [u8; 4], b: [u8; 3]| (0..3).all(|i| (a[i] as i32 - b[i] as i32).abs() <= 1);
            assert!(close(seg.segment_color(seg.labels[0] as usize), [220, 30, 30]));
            assert!(close(seg.segment_color(seg.labels[39] as usize), [20, 40, 200]));
        }
    }

    #[test]
    fn photo_palette_rounds_and_unblended_palette_truncates() {
        let img = image_from(40, 24, |x, _| if x < 20 { RED } else { BLUE });
        let photo = segment_pixels(&img, &EngineConfig::preset(ImageType::Photo, Quality::Medium));
        assert_eq!(photo.segment_color(photo.labels[0] as usize), [220, 30, 30, 255]);
        let art = segment_pixels(&img, &EngineConfig::preset(ImageType::ArtworkUnblended, Quality::Medium));
        let c = art.segment_color(art.labels[0] as usize);
        assert!(c[0] == 219 || c[0] == 220, "{c:?}");
        assert!(art.colors.is_some() && photo.colors.is_some());
    }

    #[test]
    fn fixed_palette_snaps_pixels_and_splits_components() {
        // Red | blue | red stripes with near-red noise: three components in palette colours.
        let near_red = Rgba { r: 200, g: 50, b: 40, a: 255 };
        let img = image_from(30, 10, |x, y| {
            if (x, y) == (3, 3) {
                near_red
            } else if (10..20).contains(&x) {
                BLUE
            } else {
                RED
            }
        });
        let palette = [[220, 30, 30, 255], [20, 40, 200, 255], [255, 255, 255, 255]];
        let cfg = EngineConfig::preset(ImageType::ArtworkUnblended, Quality::Medium);
        let seg = segment_pixels_fixed(&img, &cfg, &palette);
        assert_eq!(seg.segments.len(), 3);
        assert_eq!(seg.segment_color(seg.labels[0] as usize), palette[0]);
        assert_eq!(seg.segment_color(seg.labels[15] as usize), palette[1]);
        assert_eq!(seg.segment_color(seg.labels[29] as usize), palette[0]);
        assert_eq!(seg.labels[3 * 30 + 3], seg.labels[0]);
    }

    #[test]
    fn fixed_palette_anti_aliased_uses_only_palette_colours() {
        let (w, h) = (40usize, 40usize);
        let img = image_from(w, h, |x, y| {
            let d = ((x as f64 - 19.5).powi(2) + (y as f64 - 19.5).powi(2)).sqrt();
            let cover = (12.0 - d + 0.5).clamp(0.0, 1.0);
            let v = (255.0 * (1.0 - cover)).round() as u8;
            Rgba { r: v, g: v, b: v, a: 255 }
        });
        let palette = [[255, 255, 255, 255], [0, 0, 0, 255], [120, 120, 120, 255]];
        let cfg = EngineConfig::preset(ImageType::ArtworkBlended, Quality::Medium);
        let seg = segment_pixels_fixed(&img, &cfg, &palette);
        assert_eq!(seg.segments.len(), 2);
        for l in 0..seg.segments.len() {
            assert!(palette[..2].contains(&seg.segment_color(l)));
        }
    }

    #[test]
    fn disconnected_same_colour_regions_stay_separate() {
        // Two red squares on white: renumbering splits connected components.
        let img = image_from(48, 20, |x, y| {
            let in_sq = (4..14).contains(&y) && ((4..14).contains(&x) || (30..40).contains(&x));
            if in_sq { RED } else { WHITE }
        });
        let seg = segment_pixels(&img, &EngineConfig::preset(ImageType::Photo, Quality::High));
        assert_eq!(seg.segments.len(), 3);
    }

    #[test]
    fn noise_is_absorbed() {
        // A flat image with sparse single-pixel noise collapses to one segment.
        let img = image_from(32, 32, |x, y| {
            if (x * 7 + y * 13) % 37 == 0 { Rgba { r: 240, g: 240, b: 240, a: 255 } } else { WHITE }
        });
        let seg = segment_pixels(&img, &EngineConfig::preset(ImageType::Photo, Quality::Medium));
        assert_eq!(seg.segments.len(), 1);
    }
}
