//! Anti-aliased ("blended") segmentation: `SubPixelSegmenter` on top
//! of `SubPixelSegmentation`.
//!
//! Every boundary pixel is modelled as a blend of the colours of the segments around it
//! one candidate → that colour; two → a regularised blend factor; three or
//! more → a non-negative, sum-to-one least-squares fit (active set).
//! A pixel's cost is its colour residual plus penalties when the weight of its own segment is low.
//! Segments are merged greedily into the neighbour that raises the total cost
//! least, as long as the increase stays below `lambda_f`.
//!
//! A thin-segment pixel reassignment step is deliberately omitted: it would relabel each pixel to
//! candidate 0 of the last blend, which is always the pixel's own segment, so it changes nothing.

use std::collections::{BTreeSet, HashMap, HashSet};

use super::labels::{self, INV_255_F32, NBR_DX, NBR_DY, nbr_count};
use super::segmenter::{SegStats, Segmentation};
use super::shadow::{self, NUM_FEATURES};
use crate::image::RgbaImage;
use crate::params::EngineConfig;

/// Tunables of `SubPixelSegmenter` and `SubPixelSegmentation`.
#[derive(Clone, Debug, PartialEq)]
pub struct SubPixelParams {
    /// Not in the text config; the default is 20.
    pub max_iterations: i32,
    pub beach_lambda_f: f32,
    ///a small segment only takes another colour if that lowers its cost by more.
    pub color_cluster_margin: f32,
    pub lambda_f: f32,
    pub use_connected_24: bool,
    pub self_weight_eps: f32,
    pub self_weight_penalty_low: f32,
    pub self_weight_penalty_high: f32,
}

impl SubPixelParams {
    pub fn from_config(c: &EngineConfig) -> Self {
        Self {
            max_iterations: 20,
            beach_lambda_f: c.subpix_beach_lambda_f,
            color_cluster_margin: c.subpix_color_cluster_margin,
            lambda_f: c.subpix_lambda_f,
            use_connected_24: c.subpix_use_connected_24 != 0,
            self_weight_eps: c.subpixseg_self_weight_eps,
            self_weight_penalty_low: c.subpixseg_self_weight_penalty_low,
            self_weight_penalty_high: c.subpixseg_self_weight_penalty_high,
        }
    }
}

/// "No limit" size bound passed to the merge passes.
const ANY: i32 = i32::MAX;

/// Segment record.
#[derive(Clone, Debug, Default)]
struct Record {
    /// All pixels.
    n: i32,
    /// Σc and Σc² over all pixels (0..1 units).
    sum: [f32; 4],
    sumsq: [f32; 4],
    /// pixels whose whole neighbourhood is in this segment.
    n_int: i32,
    /// Σc and Σc² over interior pixels.
    sum_int: [f32; 4],
    sumsq_int: [f32; 4],
    /// boundary pixels of this segment plus the outside pixels touching it.
    set: BTreeSet<i32>,
    /// representative colour (per-channel median of the interior pixels).
    rep: [f32; 4],
    /// palette index, once the segmentation has been palettised.
    pal: Option<usize>,
}

impl Record {
    /// Add a pixel's colour to the statistics.
    fn add(&mut self, c: [f32; 4]) {
        self.n += 1;
        for i in 0..4 {
            self.sum[i] += c[i];
            self.sumsq[i] += c[i] * c[i];
        }
    }

    /// Add an interior pixel's colour to the statistics.
    fn add_interior(&mut self, c: [f32; 4]) {
        self.n_int += 1;
        for i in 0..4 {
            self.sum_int[i] += c[i];
            self.sumsq_int[i] += c[i] * c[i];
        }
    }

    /// Absorb `o`'s statistics (the pixel set is handled by the caller).
    fn absorb(&mut self, o: &Record) {
        self.n += o.n;
        self.n_int += o.n_int;
        for i in 0..4 {
            self.sum[i] += o.sum[i];
            self.sumsq[i] += o.sumsq[i];
            self.sum_int[i] += o.sum_int[i];
            self.sumsq_int[i] += o.sumsq_int[i];
        }
    }
}

/// Result of blending a pixel from its candidate segments.
struct Blend {
    color: [f32; 4],
    /// Weight of the pixel's own segment.
    alpha: f64,
}

pub struct SubPixelSegmenter<'a> {
    p: SubPixelParams,
    img: &'a RgbaImage,
    w: i32,
    h: i32,
    labels: Vec<i32>,
    parent: Vec<i32>,
    recs: Vec<Record>,
    /// Cached cost per pixel.
    cost: Vec<f32>,
    /// 24-connected mode.
    conn24: bool,
    /// palette, built from the segment colours.
    palette: Vec<[u8; 4]>,
}

impl<'a> SubPixelSegmenter<'a> {
    /// Start from the pixel segmenter's labels (output).
    pub fn new(img: &'a RgbaImage, labels: Vec<i32>, p: SubPixelParams) -> Self {
        let n = labels.iter().copied().max().map_or(0, |m| m as usize + 1);
        Self {
            p,
            img,
            w: img.width as i32,
            h: img.height as i32,
            labels,
            parent: vec![-1; n],
            recs: Vec::new(),
            cost: vec![0.0; img.width * img.height],
            conn24: false,
            palette: Vec::new(),
        }
    }

    fn px(&self, p: usize) -> [f32; 4] {
        let c = self.img.pixels[p];
        [c.r, c.g, c.b, c.a].map(|v| f32::from(v) * INV_255_F32)
    }

    fn find(&mut self, p: usize) -> i32 {
        labels::find(&mut self.labels, &mut self.parent, p)
    }

    /// Label at (x, y) with segment `from` read as `to` (if `to` ≥ 0).
    fn label_as(&mut self, x: i32, y: i32, ov: (i32, i32)) -> i32 {
        if x < 0 || y < 0 || x >= self.w || y >= self.h {
            return -1;
        }
        let l = self.find((y * self.w + x) as usize);
        if ov.0 >= 0 && l == ov.1 { ov.0 } else { l }
    }

    /// Palette colour once palettised, else the representative colour.
    fn color(&self, label: i32) -> [f32; 4] {
        let r = &self.recs[label as usize];
        match r.pal {
            Some(i) => self.palette[i].map(|v| f32::from(v) * INV_255_F32),
            None => r.rep,
        }
    }

    /// Own label first, then distinct labels of the 8 (24) neighbours, at most 9.
    fn candidates(&mut self, x: i32, y: i32, ov: (i32, i32)) -> Vec<i32> {
        let mut c = Vec::with_capacity(9);
        let own = self.label_as(x, y, ov);
        if own >= 0 {
            c.push(own);
        }
        for k in 0..nbr_count(self.conn24) {
            let l = self.label_as(x + NBR_DX[k], y + NBR_DY[k], ov);
            if l >= 0 && !c.contains(&l) {
                c.push(l);
                if c.len() > 8 {
                    break;
                }
            }
        }
        c
    }

    /// Best blend of the candidate colours for pixel (x, y).
    fn blend(&mut self, x: i32, y: i32, ov: (i32, i32)) -> Blend {
        let cands = self.candidates(x, y, ov);
        let cols: Vec<[f32; 4]> = cands.iter().map(|&l| self.color(l)).collect();
        let (w, color) = mix(&cols, self.px((y * self.w + x) as usize));
        Blend { color, alpha: w.first().copied().unwrap_or(1.0) }
    }

    /// Residual plus self-weight penalties.
    fn pixel_cost(&mut self, p: usize, ov: (i32, i32)) -> f32 {
        let (x, y) = (p as i32 % self.w, p as i32 / self.w);
        let b = self.blend(x, y, ov);
        let c = self.px(p);
        let d = [0, 1, 2, 3].map(|i| c[i] - b.color[i]);
        let resid = d[0] * d[0] + d[1] * d[1] + d[2] * d[2] + d[3] * d[3];
        let mut pen = 0.0f64;
        if 1.0 - b.alpha > 0.0 {
            pen = f64::from(self.p.self_weight_penalty_low) * (1.0 - b.alpha);
        }
        let short = f64::from(self.p.self_weight_eps) - b.alpha;
        if short > 0.0 {
            pen += f64::from(self.p.self_weight_penalty_high) * short;
        }
        (pen + f64::from(resid)) as f32
    }

    /// Squared error of `a`'s interior pixels against the colour of `as_b`
    /// (or `a`'s own colour when `as_b` < 0).
    fn interior_cost(&self, a: i32, as_b: i32) -> f64 {
        let r = &self.recs[a as usize];
        if r.n_int < 1 {
            return 0.0;
        }
        let col = self.color(if as_b >= 0 { as_b } else { a });
        let n = f64::from(r.n_int);
        let inv = 1.0 / n;
        let mut total = 0.0;
        for ch in 0..4 {
            let mu = f64::from(r.sum_int[ch]) * inv;
            let mut var = f64::from(r.sumsq_int[ch]) - n * mu * mu;
            // Channels 0..2 are clamped at zero; the alpha channel is not.
            if ch < 3 && var <= 0.0 {
                var = 0.0;
            }
            let e = f64::from(col[ch]) - mu;
            total += e * e * n + var;
        }
        total
    }

    /// Pixels of `a`'s set, in ascending order.
    fn set_of(&self, a: i32) -> Vec<i32> {
        self.recs[a as usize].set.iter().copied().collect()
    }

    /// Total cost of `a`'s set. Cached values, or recomputed with `a` read as
    /// `as_b` (optionally storing the new values).
    fn boundary_cost(&mut self, cached: bool, store: bool, a: i32, as_b: i32) -> f64 {
        let pixels = self.set_of(a);
        if cached {
            return pixels.iter().map(|&p| f64::from(self.cost[p as usize])).sum();
        }
        let mut acc = 0.0f32;
        for p in pixels {
            let c = self.pixel_cost(p as usize, (as_b, a));
            if store {
                self.cost[p as usize] = c;
            }
            acc += c;
        }
        f64::from(acc)
    }

    /// Cost increase of merging `a` into `b` (early out above λ).
    fn merge_cost(&mut self, b: i32, a: i32) -> f64 {
        let own = self.interior_cost(a, -1) + self.boundary_cost(true, false, a, -1);
        let mut d = self.interior_cost(a, b) - own;
        if d <= f64::from(self.p.lambda_f) {
            d += self.boundary_cost(false, false, a, b);
        }
        d
    }

    /// Pixel `p` belongs to `b` and so do all its in-image neighbours.
    fn is_interior(&mut self, b: i32, p: i32) -> bool {
        if self.find(p as usize) != b {
            return false;
        }
        let (x, y) = (p % self.w, p / self.w);
        for k in 0..nbr_count(self.conn24) {
            let (nx, ny) = (x + NBR_DX[k], y + NBR_DY[k]);
            if nx >= 0 && nx < self.w && ny >= 0 && ny < self.h && self.find((ny * self.w + nx) as usize) != b {
                return false;
            }
        }
        true
    }

    /// Merge `a` into `b`.
    fn merge(&mut self, b: i32, a: i32) {
        self.parent[a as usize] = b;
        let ra = std::mem::take(&mut self.recs[a as usize]);
        self.recs[b as usize].absorb(&ra);
        // Recompute and store the costs of a's pixels; a still owns its set at this point.
        self.recs[a as usize].set = ra.set;
        self.boundary_cost(false, true, a, b);
        let pixels = std::mem::take(&mut self.recs[a as usize].set);
        for p in pixels {
            if self.recs[b as usize].set.insert(p) {
                continue;
            }
            if self.is_interior(b, p) {
                let c = self.px(p as usize);
                let rb = &mut self.recs[b as usize];
                rb.add_interior(c);
                rb.set.remove(&p);
            }
        }
    }

    /// One greedy merge pass. Segments of size ≤ `max_self` merge into the
    /// cheapest neighbour of size ≤ `max_other` if the cost increase is below λ.
    fn merge_pass(&mut self, max_other: i32, max_self: i32) -> i32 {
        let mut merged = 0;
        // Pairs already evaluated in this pass, keyed `a << 16 | b`.
        let mut tried: HashSet<u32> = HashSet::new();
        for a in 0..self.recs.len() as i32 {
            let size = if self.conn24 { self.recs[a as usize].n_int } else { self.recs[a as usize].n };
            if self.parent[a as usize] >= 0 || size > max_self {
                continue;
            }
            let mut best = f64::from(self.p.lambda_f);
            let mut best_b = -1;
            for p in self.set_of(a) {
                let b = self.find(p as usize);
                if b == a {
                    continue;
                }
                let key = ((a as u32) << 16) | b as u32;
                if tried.contains(&key) || self.recs[b as usize].n > max_other {
                    continue;
                }
                let c = self.merge_cost(b, a);
                if c < best {
                    best = c;
                    best_b = b;
                }
                tried.insert(key);
            }
            if best_b >= 0 {
                self.merge(best_b, a);
                merged += 1;
            }
        }
        merged
    }

    /// Renumber, then rebuild records, pixel sets, representative colours and
    /// cached pixel costs.
    fn rebuild(&mut self) {
        let n = (self.w * self.h) as usize;
        let roots: Vec<i32> = (0..n).map(|p| self.find(p)).collect();
        let (labels, count) = labels::relabel_components(self.w, self.h, &roots);
        // carries each component's palette index over from its old segment.
        let mut pal = vec![None; count];
        for (&new, &old) in labels.iter().zip(&roots) {
            pal[new as usize] = self.recs.get(old as usize).and_then(|r| r.pal);
        }
        self.labels = labels;
        self.parent = vec![-1; count];
        self.recs = pal.into_iter().map(|pal| Record { pal, ..Record::default() }).collect();
        let mut hist: Vec<[Vec<u8>; 4]> = vec![Default::default(); count];
        let nn = nbr_count(self.conn24);
        for y in 0..self.h {
            for x in 0..self.w {
                let p = (y * self.w + x) as usize;
                let a = self.labels[p];
                let mut boundary = false;
                for k in 0..nn {
                    let (nx, ny) = (x + NBR_DX[k], y + NBR_DY[k]);
                    if nx < 0 || ny < 0 || nx >= self.w || ny >= self.h {
                        continue;
                    }
                    let q = ny * self.w + nx;
                    if self.labels[q as usize] != a {
                        boundary = true;
                        self.recs[a as usize].set.insert(q);
                    }
                }
                let c = self.px(p);
                let r = &mut self.recs[a as usize];
                r.add(c);
                if boundary {
                    r.set.insert(p as i32);
                } else {
                    r.add_interior(c);
                    let raw = self.img.pixels[p];
                    for (h, v) in hist[a as usize].iter_mut().zip([raw.r, raw.g, raw.b, raw.a]) {
                        h.push(v);
                    }
                }
            }
        }
        for (r, h) in self.recs.iter_mut().zip(hist.iter_mut()) {
            r.rep = if r.n_int < 3 {
                let s = 1.0 / r.n as f32;
                r.sum.map(|v| v * s)
            } else {
                // Nth_element at n/2 per channel.
                let mut m = [0.0f32; 4];
                for (ch, v) in h.iter_mut().enumerate() {
                    let mid = v.len() / 2;
                    let (_, &mut med, _) = v.select_nth_unstable(mid);
                    m[ch] = f32::from(med) * INV_255_F32;
                }
                m
            };
        }
        // Cache the cost of each segment's own boundary pixels.
        for a in 0..count as i32 {
            for p in self.set_of(a) {
                if self.find(p as usize) == a {
                    self.cost[p as usize] = self.pixel_cost(p as usize, (-1, -1));
                }
            }
        }
    }

    /// Repeat `merge_pass` until nothing merges, at most `max_iterations` more times.
    fn passes(&mut self, first: (i32, i32), next: impl Fn(i32) -> (i32, i32)) {
        let mut r = self.merge_pass(first.0, first.1);
        let mut i = 0;
        while r != 0 && i < self.p.max_iterations {
            let (o, s) = next(i);
            r = self.merge_pass(o, s);
            i += 1;
        }
    }

    /// Run the merge passes.
    pub fn run(&mut self) {
        self.conn24 = false;
        self.rebuild();
        self.passes((1, 1), |i| (i + 2, i + 2));
        self.rebuild();
        self.passes((ANY, 1), |i| (ANY, i + 2));
        self.passes((ANY, ANY), |_| (ANY, ANY));
        if self.p.use_connected_24 {
            self.conn24 = true;
            self.rebuild();
            let m = self.p.max_iterations;
            self.passes((ANY, m), |_| (ANY, m));
            self.conn24 = false;
            self.rebuild();
        }
        labels::fix_diagonals(self.img, &mut self.labels, &mut self.parent);
        self.rebuild();
        self.passes((ANY, 4), |_| (ANY, 4));
        self.rebuild();
    }

    /// Give every segment with fewer than 3 interior pixels the colour of the
    /// larger segment that explains its pixels best, if that beats its own cost by
    /// `color_cluster_margin`. Only colours change; `renumber` then merges equal colours.
    pub fn reject_small(&mut self) {
        // Ordered set of (interior count, label), flattened in that order.
        let big: Vec<i32> = (0..self.recs.len() as i32)
            .filter(|&l| self.recs[l as usize].n_int > 2)
            .map(|l| (self.recs[l as usize].n_int, l))
            .collect::<BTreeSet<_>>()
            .into_iter()
            .map(|(_, l)| l)
            .collect();
        for s in 0..self.recs.len() as i32 {
            if self.recs[s as usize].n_int >= 3 {
                continue;
            }
            // With as_b = −1: own interior + cached boundary cost.
            let own = self.interior_cost(s, -1) + self.boundary_cost(true, false, s, -1);
            let mut best = own - f64::from(self.p.color_cluster_margin);
            let mut pick = s;
            for &b in &big {
                let f = self.interior_cost(s, b);
                if f < best {
                    let g = self.bounded_cost(best - f, s, b);
                    if f + g < best {
                        best = f + g;
                        pick = b;
                    }
                }
            }
            if pick != s {
                let (rep, pal) = (self.recs[pick as usize].rep, self.recs[pick as usize].pal);
                let r = &mut self.recs[s as usize];
                r.rep = rep;
                r.pal = pal;
            }
        }
    }

    /// Cost of `a`'s set with `a` read as `b`, summed in f64 and stopped as soon
    /// as it exceeds `bound`.
    fn bounded_cost(&mut self, bound: f64, a: i32, b: i32) -> f64 {
        let mut acc = 0.0f64;
        for p in self.set_of(a) {
            acc += f64::from(self.pixel_cost(p as usize, (b, a)));
            if bound < acc {
                break;
            }
        }
        acc
    }

    /// `generatePaletteFromSegments` maps each segment colour
    /// (rounded to bytes, deduplicated in label order) to a palette index; pixels are relabelled
    /// by palette index and split into 4-connected components, which are then rebuilt.
    pub fn renumber(&mut self) {
        // Segment colours may still refer to the old palette (fixed-palette mode).
        let colors: Vec<[u8; 4]> = (0..self.recs.len() as i32).map(|l| self.color(l).map(to_byte)).collect();
        self.palette.clear();
        let mut index = HashMap::new();
        let mut seg_pal = Vec::with_capacity(self.recs.len());
        for c in colors {
            let next = self.palette.len();
            let i = *index.entry(c).or_insert(next);
            if i == next {
                self.palette.push(c);
            }
            seg_pal.push(i);
        }
        let n = (self.w * self.h) as usize;
        for p in 0..n {
            let l = self.find(p) as usize;
            self.labels[p] = seg_pal[l] as i32;
        }
        self.parent = vec![-1; self.palette.len()];
        self.recs = (0..self.palette.len()).map(|i| Record { pal: Some(i), ..Record::default() }).collect();
        self.conn24 = false;
        self.rebuild();
    }

    /// Drop the palette, rebuild the 8-connected
    /// statistics, compute the shadow features of every segment and classify them.
    /// Returns the shadow flag and feature vector of each segment.
    fn flag_shadows(&mut self) -> (Vec<bool>, Vec<[f64; NUM_FEATURES]>) {
        for r in &mut self.recs {
            r.pal = None;
        }
        self.conn24 = false;
        self.rebuild();
        let n = self.recs.len();
        let np = (self.w * self.h) as usize;
        let (plane_a, plane_b) = shadow::edge_planes(self.img);
        let mut f = vec![[0.0f64; NUM_FEATURES]; n];
        let mut sum_a = vec![0.0f32; n];
        let mut sum_b = vec![0.0f32; n];
        // Σ c·cᵀ per segment, element [i][j] = (cᵢ · byteⱼ) / 255.
        let mut cov = vec![[0.0f32; 16]; n];
        for p in 0..np {
            let l = self.find(p) as usize;
            sum_a[l] += plane_a[p];
            sum_b[l] += plane_b[p];
            let raw = self.img.pixels[p];
            let bytes = [raw.r, raw.g, raw.b, raw.a].map(f32::from);
            let c = bytes.map(|v| v * INV_255_F32);
            for i in 0..4 {
                for j in 0..4 {
                    cov[l][i * 4 + j] += c[i] * bytes[j] * INV_255_F32;
                }
            }
        }
        // Covariance and its principal axis.
        let mut means = vec![[0.0f32; 4]; n];
        let mut axes = vec![[0.0f32; 4]; n];
        for l in 0..n {
            let r = &self.recs[l];
            let inv = 1.0 / r.n as f32;
            let mean = r.sum.map(|v| v * inv);
            for i in 0..4 {
                for j in 0..4 {
                    cov[l][i * 4 + j] = inv * cov[l][i * 4 + j] - mean[j] * mean[i];
                }
            }
            means[l] = mean;
            axes[l] = shadow::principal_axis(&cov[l]);
        }
        // Feature 41: residual of every pixel after projecting onto its segment's axis.
        let mut resid = vec![0.0f32; n];
        for p in 0..np {
            let l = self.find(p) as usize;
            let (m, v) = (means[l], axes[l]);
            let c = self.px(p);
            let d = [0, 1, 2, 3].map(|i| c[i] - m[i]);
            let t = d[0] * v[0] + v[1] * d[1] + v[2] * d[2] + v[3] * d[3];
            let e = [0, 1, 2, 3].map(|i| c[i] - (v[i] * t + m[i]));
            resid[l] += e[0] * e[0] + e[1] * e[1] + e[2] * e[2] + e[3] * e[3];
        }
        let colors: Vec<[f32; 4]> = (0..n as i32).map(|l| self.color(l)).collect();
        let mut flags = vec![false; n];
        for l in 0..n {
            let r = &self.recs[l];
            let fl = &mut f[l];
            let cnt = f64::from(r.n);
            fl[0] = f64::from(sum_a[l]);
            fl[1] = f64::from(sum_b[l]);
            fl[3] = fl[1].sqrt();
            fl[4] = (fl[0] / cnt).sqrt();
            fl[5] = (fl[1] / cnt).sqrt();
            fl[8] = cnt;
            fl[0x1c] = fl[3] / cnt;
            fl[0x29] = f64::from(resid[l]);
            // Closest representative colour among segments with more than 10 interior pixels.
            fl[0x27] = 1e100;
            for o in &self.recs {
                if o.n_int > 10 {
                    let d = [0, 1, 2, 3].map(|i| o.rep[i] - r.rep[i]);
                    let d2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2] + d[3] * d[3];
                    if d2 < fl[0x27] as f32 {
                        fl[0x27] = f64::from(d2);
                    }
                }
            }
            let own_luma = shadow::luma(colors[l]) as f32;
            fl[0x15] = f64::from(own_luma);
            let inv = 1.0 / r.n as f32;
            let var: [f32; 4] = [0, 1, 2, 3].map(|i| {
                let m = r.sum[i] * inv;
                r.sumsq[i] * inv - m * m
            });
            fl[0x16] = f64::from(var[0] + var[1] + var[2] + var[3]);
            fl[0x17] = f64::from(r.n_int) / r.set.len() as f64;
            // Segments touching this one: labels of the pixels in its set (itself included).
            let mut touching = BTreeSet::new();
            for &q in &r.set {
                touching.insert(self.labels[q as usize]);
            }
            fl[0xb] = touching.len() as f64 - 1.0;
            for &k in &touching {
                if k as usize == l {
                    continue;
                }
                let o = &self.recs[k as usize];
                if o.n > 9 {
                    fl[0xe] += 1.0;
                }
                if o.n_int > 14 {
                    fl[0x12] += 1.0;
                }
                if shadow::luma(colors[k as usize]) <= f64::from(own_luma) {
                    fl[0x14] += 1.0;
                } else {
                    fl[0x13] += 1.0;
                }
            }
            flags[l] = shadow::is_shadow(fl);
        }
        (flags, f)
    }

    /// The 4-neighbour of `p` (left, right, up, down) outside any shadow whose
    /// pixel colour is closest to `p`'s, with its label. `None` if every neighbour is a shadow.
    fn nearest_non_shadow(&self, p: usize, shadow: &[bool]) -> Option<(f32, i32)> {
        let (x, y) = (p as i32 % self.w, p as i32 / self.w);
        let c = self.px(p);
        let mut best = 1e10f32;
        let mut label = None;
        for (dx, dy) in [(-1, 0), (1, 0), (0, -1), (0, 1)] {
            let (nx, ny) = (x + dx, y + dy);
            if nx < 0 || ny < 0 || nx >= self.w || ny >= self.h {
                continue;
            }
            let q = (ny * self.w + nx) as usize;
            let l = self.labels[q];
            if shadow[l as usize] {
                continue;
            }
            let o = self.px(q);
            let d = [0, 1, 2, 3].map(|i| o[i] - c[i]);
            let d2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2] + d[3] * d[3];
            if best > d2 {
                best = d2;
                label = Some(l);
            }
        }
        label.map(|l| (best, l))
    }

    /// Hand every shadow's pixels to the neighbouring segments, cheapest pixel
    /// first, then keep or undo each shadow's dissolution with the tree, which
    /// compares the pixel costs around the shadow before and after.
    fn dissolve_shadows(&mut self, shadow: &[bool], feats: &[[f64; NUM_FEATURES]]) {
        let ids: Vec<i32> = (0..shadow.len() as i32).filter(|&l| shadow[l as usize]).collect();
        if ids.is_empty() {
            return;
        }
        let k = ids.len();
        // The map is already compact.
        self.conn24 = true;
        self.rebuild();
        let mut cost0 = vec![0.0f32; k];
        let mut outside: Vec<Vec<i32>> = vec![Vec::new(); k];
        for (i, &l) in ids.iter().enumerate() {
            // Quirk: the shadow's index, not its label, is passed to the interior cost.
            cost0[i] = self.interior_cost(i as i32, -1) as f32;
            for &q in &self.recs[l as usize].set {
                cost0[i] += self.cost[q as usize];
                if self.labels[q as usize] != l {
                    outside[i].push(q);
                }
            }
        }
        let index: Vec<Option<usize>> = {
            let mut v = vec![None; shadow.len()];
            for (i, &l) in ids.iter().enumerate() {
                v[l as usize] = Some(i);
            }
            v
        };
        let mut pixels: Vec<Vec<usize>> = vec![Vec::new(); k];
        for p in 0..(self.w * self.h) as usize {
            if let Some(i) = index[self.labels[p] as usize] {
                pixels[i].push(p);
            }
        }
        for px in &pixels {
            let mut work = px.clone();
            while !work.is_empty() {
                let mut best: Option<(usize, f32, i32)> = None;
                for (j, &p) in work.iter().enumerate() {
                    if let Some((c, l)) = self.nearest_non_shadow(p, shadow) {
                        if best.is_none_or(|b| b.1 > c) {
                            best = Some((j, c, l));
                        }
                    }
                }
                // This cannot make progress;
                // stop and leave the remaining pixels in the shadow.
                let Some((j, _, l)) = best else { break };
                self.labels[work[j]] = l;
                work.remove(j);
            }
        }
        // Cost of the same pixels after dissolving: relabel + init (8-connected), then a
        // 24-connected rebuild.
        let dissolved = self.labels.clone();
        self.conn24 = false;
        self.rebuild();
        self.conn24 = true;
        self.rebuild();
        let mut cost1 = vec![0.0f32; k];
        for i in 0..k {
            let all: Vec<usize> = pixels[i].iter().copied().chain(outside[i].iter().map(|&q| q as usize)).collect();
            for p in all {
                cost1[i] = (f64::from(self.pixel_cost(p, (-1, -1))) + f64::from(cost1[i])) as f32;
            }
        }
        // Back to the dissolved map in the old numbering.
        self.labels = dissolved;
        self.parent = vec![-1; shadow.len()];
        for i in 0..k {
            let mut f = feats[ids[i] as usize];
            let m = (pixels[i].len() + outside[i].len()) as f32;
            let inv = f64::from(1.0 / m);
            let (c0, c1) = (f64::from(cost0[i]), f64::from(cost1[i]));
            f[0x21] = c0;
            f[0x22] = c1;
            f[0x23] = inv * c0;
            f[0x24] = inv * c1;
            f[0x25] = c1 - c0;
            f[0x26] = f[0x25] / f64::from(m);
            if !shadow::keep_dissolved(&f, 0) {
                for &p in &pixels[i] {
                    self.labels[p] = ids[i];
                }
            }
        }
        // + init (0x2b, 0).
        self.conn24 = false;
        self.rebuild();
    }

    /// Post-processing for the flexible-palette anti-aliased case with the default flags.
    /// Returns the number of segments classified as shadows.
    pub fn finish(&mut self) -> usize {
        self.reject_small();
        self.renumber();
        let (flags, feats) = self.flag_shadows();
        self.dissolve_shadows(&flags, &feats);
        let shadows = flags.iter().filter(|&&s| s).count();
        self.reject_small();
        self.renumber();
        shadows
    }

    /// Fixed-palette tail: segments snap to the nearest palette colour instead of the first
    /// `reject_small`. `palette` is RGBA.
    pub fn finish_fixed(&mut self, palette: &[[u8; 4]]) -> usize {
        self.palette = palette.to_vec();
        self.palettize(0);
        self.renumber();
        let (flags, feats) = self.flag_shadows();
        self.dissolve_shadows(&flags, &feats);
        let shadows = flags.iter().filter(|&&s| s).count();
        self.reject_small();
        self.palettize(0);
        self.renumber();
        shadows
    }

    /// Every segment reached from a pixel that has no palette index yet and at
    /// least `min` interior pixels takes the palette colour nearest its representative colour
    /// (squared RGBA distance; first wins ties).
    fn palettize(&mut self, min_interior: i32) {
        if self.palette.is_empty() {
            return;
        }
        let pal: Vec<[f32; 4]> = self.palette.iter().map(|c| c.map(|v| f32::from(v) * INV_255_F32)).collect();
        for p in 0..(self.w * self.h) as usize {
            let l = self.find(p) as usize;
            let r = &mut self.recs[l];
            if r.pal.is_some() || r.n_int < min_interior {
                continue;
            }
            let dist =
                |q: &[f32; 4]| -> f64 { (0..4).map(|i| f64::from(r.rep[i] - q[i])).fold(0.0, |acc, d| acc + d * d) };
            let mut best = 0;
            let mut best_d = dist(&pal[0]);
            for (j, q) in pal.iter().enumerate().skip(1) {
                let d = dist(q);
                if d < best_d {
                    best = j;
                    best_d = d;
                }
            }
            r.rep = pal[best];
            r.pal = Some(best);
        }
    }

    /// Final labels with each segment's palette colour.
    pub fn into_segmentation(self) -> Segmentation {
        let segments = self.recs.iter().map(|r| SegStats { count: r.n, sum: r.sum }).collect();
        let colors = (0..self.recs.len() as i32).map(|l| self.color(l).map(to_byte)).collect();
        Segmentation {
            width: self.w as usize,
            height: self.h as usize,
            labels: self.labels.iter().map(|&l| l as u32).collect(),
            segments,
            colors: Some(colors),
        }
    }
}

/// Weights of the candidate colours `cols` (own segment first) that best
/// explain pixel colour `c`, and the blended colour.
/// One candidate: weight 1. Two: regularised projection onto the segment between them. More:
/// regularised least squares on RGB with Σw = 1 and an active set for negative weights.
pub(crate) fn mix(cols: &[[f32; 4]], c: [f32; 4]) -> (Vec<f64>, [f32; 4]) {
    match cols.len() {
        0 => (vec![1.0], [0.0; 4]),
        1 => (vec![1.0], cols[0]),
        2 => {
            let (c0, c1) = (cols[0], cols[1]);
            let d = [0, 1, 2, 3].map(|i| c1[i] - c0[i]);
            let dd = f64::from(d[0] * d[0] + d[1] * d[1] + d[2] * d[2] + d[3] * d[3]);
            if dd <= 0.0 {
                return (vec![0.5, 0.5], c0);
            }
            let dot: f64 = (0..4).map(|i| f64::from(d[i]) * f64::from(c[i] - c0[i])).sum();
            let mut t = dot / dd;
            t += (1.0 - 2.0 * t) * (0.01 / (dd + 0.02));
            let t = t.clamp(0.0, 1.0);
            let tf = t as f32;
            (vec![1.0 - t, t], [0, 1, 2, 3].map(|i| c0[i] + tf * d[i]))
        }
        k => {
            let cols: Vec<[f64; 3]> =
                cols.iter().map(|v| [f64::from(v[0]), f64::from(v[1]), f64::from(v[2])]).collect();
            let target = [f64::from(c[0]), f64::from(c[1]), f64::from(c[2])];
            let w = simplex_lsq(&cols, target);
            let mut out = [0.0f64; 3];
            for j in 0..k {
                for ch in 0..3 {
                    out[ch] += w[j] * cols[j][ch];
                }
            }
            (w, [out[0] as f32, out[1] as f32, out[2] as f32, c[3]])
        }
    }
}

/// `(int)(v·255 + 0.5)`.
fn to_byte(v: f32) -> u8 {
    (f64::from(v) * 255.0 + 0.5) as u8
}

/// Weights `w` minimising `‖target − Σ wⱼ colsⱼ‖² + 0.01‖w‖²`
/// subject to `Σw = 1`, with an active set that pins the most negative weight (< −1e−4) to 0.
fn simplex_lsq(cols: &[[f64; 3]], target: [f64; 3]) -> Vec<f64> {
    let k = cols.len();
    let mut a = vec![0.0; k * k];
    let mut b = vec![0.0; k];
    for i in 0..k {
        for j in 0..k {
            a[i * k + j] = (0..3).map(|ch| cols[i][ch] * cols[j][ch]).sum::<f64>();
        }
        a[i * k + i] += 0.01;
        b[i] = (0..3).map(|ch| cols[i][ch] * target[ch]).sum();
    }
    let mut active: Vec<usize> = Vec::new();
    loop {
        // Constraint rows: Σw = 1, then w_i = 0 for each active i.
        let m = 1 + active.len();
        let row = |r: usize, j: usize| {
            if r == 0 {
                1.0
            } else if active[r - 1] == j {
                1.0
            } else {
                0.0
            }
        };
        let ainv_b = solve(k, &a, &b);
        // X = A⁻¹ Bᵀ (k × m), S = B X (m × m), λ = S⁻¹ (B A⁻¹ b − d).
        let cols_x: Vec<Vec<f64>> =
            (0..m).map(|r| solve(k, &a, &(0..k).map(|j| row(r, j)).collect::<Vec<_>>())).collect();
        let mut s = vec![0.0; m * m];
        let mut rhs = vec![0.0; m];
        for r in 0..m {
            for c in 0..m {
                s[r * m + c] = (0..k).map(|j| row(r, j) * cols_x[c][j]).sum();
            }
            rhs[r] = (0..k).map(|j| row(r, j) * ainv_b[j]).sum::<f64>() - if r == 0 { 1.0 } else { 0.0 };
        }
        let lambda = solve(m, &s, &rhs);
        let w: Vec<f64> = (0..k).map(|j| ainv_b[j] - (0..m).map(|r| cols_x[r][j] * lambda[r]).sum::<f64>()).collect();
        let mut most = 0.0;
        let mut idx = None;
        for (j, &v) in w.iter().enumerate() {
            if v < most && !active.contains(&j) {
                most = v;
                idx = Some(j);
            }
        }
        match idx {
            Some(j) if w[j] < -0.0001 && active.len() + 1 < k => active.push(j),
            _ => return w,
        }
    }
}

/// Solve the symmetric positive-definite system `m x = rhs` (Cholesky).
fn solve(n: usize, m: &[f64], rhs: &[f64]) -> Vec<f64> {
    let mut l = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..=i {
            let mut s = m[i * n + j];
            for k in 0..j {
                s -= l[i * n + k] * l[j * n + k];
            }
            if i == j {
                l[i * n + i] = if s > 0.0 { s.sqrt() } else { f64::MIN_POSITIVE };
            } else {
                l[i * n + j] = s / l[j * n + j];
            }
        }
    }
    let mut y = vec![0.0; n];
    for i in 0..n {
        let s: f64 = (0..i).map(|k| l[i * n + k] * y[k]).sum();
        y[i] = (rhs[i] - s) / l[i * n + i];
    }
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let s: f64 = (i + 1..n).map(|k| l[k * n + i] * x[k]).sum();
        x[i] = (y[i] - s) / l[i * n + i];
    }
    x
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Rgba;
    use crate::params::{ImageType, Quality};

    #[test]
    fn simplex_lsq_recovers_a_mixture() {
        let cols = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let w = simplex_lsq(&cols, [0.5, 0.3, 0.2]);
        assert!((w.iter().sum::<f64>() - 1.0).abs() < 1e-9);
        assert!((w[0] - 0.5).abs() < 0.02 && (w[1] - 0.3).abs() < 0.02, "{w:?}");
    }

    #[test]
    fn simplex_lsq_pins_negative_weights() {
        // Target outside the hull of the first two colours on the side of the first.
        let cols = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let w = simplex_lsq(&cols, [1.2, 0.0, -0.2]);
        assert!(w.iter().all(|&v| v >= -1e-4), "{w:?}");
    }

    #[test]
    fn anti_aliased_disc_is_one_segment_on_background() {
        // A black disc on white with a one-pixel anti-aliased rim.
        let (w, h) = (40usize, 40usize);
        let mut px = Vec::with_capacity(w * h);
        for y in 0..h {
            for x in 0..w {
                let d = ((x as f64 - 19.5).powi(2) + (y as f64 - 19.5).powi(2)).sqrt();
                let cover = (12.0 - d + 0.5).clamp(0.0, 1.0);
                let v = (255.0 * (1.0 - cover)).round() as u8;
                px.push(Rgba { r: v, g: v, b: v, a: 255 });
            }
        }
        let img = RgbaImage::new(w, h, px);
        let seg = crate::stages::segmenter::segment_pixels(
            &img,
            &EngineConfig::preset(ImageType::ArtworkBlended, Quality::Medium),
        );
        assert_eq!(seg.segments.len(), 2, "rim pixels must not form their own segments");
    }
}
