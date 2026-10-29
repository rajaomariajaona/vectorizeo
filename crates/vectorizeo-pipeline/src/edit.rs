//! Manual edit pages: Edit Segmentation and Remove Background.
//!
//! The segmentation editor works on a colour per pixel (the edit image, not
//! on labels; [`SegEditor::image`] becomes a segmentation again when committed
//! ([`vectorizeo_core::stages::segmenter::segmentation_from_colors`]). History is an operation log
//! replayed onto the image fetched from the segmentation.

use std::collections::BTreeSet;
use std::sync::Arc;

use vectorizeo_core::RgbaImage;
use vectorizeo_core::stages::segmenter::Segmentation;

use crate::Flatten;

/// One recorded edit (`ToolActionRecord`, 0x40 bytes). Coordinates are image pixels.
#[derive(Clone, Debug, PartialEq)]
pub enum Action {
    /// Type 3: a stroke piece from the previous pointer position to the current one.
    Pencil { from: [f64; 2], to: [f64; 2], color: [u8; 4] },
    /// Type 5.
    Fill { x: i32, y: i32, color: [u8; 4] },
    /// Type 6.
    Zap { x: i32, y: i32 },
}

/// Packed 0xAARRGGBB order of a colour: Zap's neighbour set is ordered by it.
fn argb(c: [u8; 4]) -> u32 {
    u32::from(c[3]) << 24 | u32::from(c[0]) << 16 | u32::from(c[1]) << 8 | u32::from(c[2])
}

/// Neighbour order of the flood fill.
const FILL_DX: [i32; 4] = [-1, 1, 0, 0];
const FILL_DY: [i32; 4] = [0, 0, -1, 1];

/// Zap's reference colours: the input with flatten modes 1 and 2 composited
/// over the flatten colour, `a' = a + (1−a)·bg_a`, `c' = c·a + (1−a)·bg_c·bg_a`, bytes truncated.
/// `loaded` is premultiplied, so `c·a` is its byte / 255; mode 3 uses the premultiplied bytes.
pub fn zap_reference(loaded: &RgbaImage, flatten: Flatten) -> Vec<[u8; 4]> {
    loaded
        .pixels
        .iter()
        .map(|p| {
            let c = [p.r, p.g, p.b, p.a];
            let Flatten(Some(bg)) = flatten else { return c };
            let a = f64::from(p.a) / 255.0;
            let bga = f64::from(bg[3]);
            let ch = |i: usize| f64::from(c[i]) / 255.0 + (1.0 - a) * f64::from(bg[i]) * bga;
            let v = [ch(0), ch(1), ch(2), a + (1.0 - a) * bga];
            v.map(|v| (v * 255.0) as i32 as u8)
        })
        .collect()
}

/// Edit Segmentation state: the image fetched from a segmentation, the
/// edited image, and undo/redo lists of actions.
#[derive(Clone, Debug)]
pub struct SegEditor {
    pub width: usize,
    pub height: usize,
    base: Arc<Vec<[u8; 4]>>,
    image: Arc<Vec<[u8; 4]>>,
    /// Zap compares with these instead of the segment colours.
    reference: Arc<Vec<[u8; 4]>>,
    undo: Vec<Action>,
    redo: Vec<Action>,
    revision: u64,
}

impl SegEditor {
    /// Fetch: every pixel in its segment's output colour, empty history.
    pub fn new(seg: &Segmentation, reference: Vec<[u8; 4]>) -> Self {
        let colors: Vec<[u8; 4]> = (0..seg.segments.len()).map(|k| seg.segment_color(k)).collect();
        let base = Arc::new(seg.labels.iter().map(|&l| colors[l as usize]).collect::<Vec<_>>());
        Self {
            width: seg.width,
            height: seg.height,
            image: base.clone(),
            base,
            reference: Arc::new(reference),
            undo: Vec::new(),
            redo: Vec::new(),
            revision: 0,
        }
    }

    /// The edited colour image, row by row.
    pub fn image(&self) -> &Arc<Vec<[u8; 4]>> {
        &self.image
    }

    /// Changes whenever the image does (undo/redo/revert included).
    pub fn revision(&self) -> u64 {
        self.revision
    }

    pub fn has_edits(&self) -> bool {
        !self.undo.is_empty()
    }

    pub fn can_undo(&self) -> bool {
        !self.undo.is_empty()
    }

    pub fn can_redo(&self) -> bool {
        !self.redo.is_empty()
    }

    /// The colour at an image pixel (the colour picker reads the edit image,).
    pub fn color_at(&self, x: i32, y: i32) -> Option<[u8; 4]> {
        self.index(x, y).map(|i| self.image[i])
    }

    fn index(&self, x: i32, y: i32) -> Option<usize> {
        (x >= 0 && y >= 0 && (x as usize) < self.width && (y as usize) < self.height)
            .then(|| y as usize * self.width + x as usize)
    }

    /// `takeAction(rec, add = true)`: apply, and record it if it changed anything (
    /// clears the redo list).
    pub fn apply(&mut self, a: Action) -> bool {
        if !self.run(&a) {
            return false;
        }
        self.undo.push(a);
        self.redo.clear();
        self.revision += 1;
        true
    }

    /// Move the last action to the redo list, restore the fetched image and
    /// replay the rest.
    pub fn undo(&mut self) {
        let Some(a) = self.undo.pop() else { return };
        self.redo.push(a);
        self.replay();
    }

    ///.
    pub fn redo(&mut self) {
        let Some(a) = self.redo.pop() else { return };
        self.run(&a);
        self.undo.push(a);
        self.revision += 1;
    }

    /// Back to the fetched image; every action (undone ones after the rest) can
    /// be redone in order.
    pub fn revert(&mut self) {
        if self.undo.is_empty() {
            return;
        }
        // `redo` is a stack (last = next to redo); the full history is undo ++ redo in order.
        let mut all = std::mem::take(&mut self.undo);
        all.extend(self.redo.drain(..).rev());
        self.redo = all.into_iter().rev().collect();
        self.image = self.base.clone();
        self.revision += 1;
    }

    fn replay(&mut self) {
        self.image = self.base.clone();
        for a in std::mem::take(&mut self.undo).iter() {
            self.run(a);
            self.undo.push(a.clone());
        }
        self.revision += 1;
    }

    fn run(&mut self, a: &Action) -> bool {
        match *a {
            Action::Pencil { from, to, color } => self.pencil(from, to, color),
            Action::Fill { x, y, color } => self.fill(x, y, color, false).is_some(),
            Action::Zap { x, y } => self.zap(x, y),
        }
    }

    /// → `setSegPixel`.
    fn set(&mut self, i: usize, c: [u8; 4]) -> bool {
        if self.image[i] == c {
            return false;
        }
        Arc::make_mut(&mut self.image)[i] = c;
        true
    }

    /// A DDA with four samples per pixel of stroke length; both ends must be
    /// inside the image.
    fn pencil(&mut self, from: [f64; 2], to: [f64; 2], color: [u8; 4]) -> bool {
        let (w, h) = (self.width as f64, self.height as f64);
        let inside = |p: [f64; 2]| p[0] >= 0.0 && p[0] < w && p[1] >= 0.0 && p[1] < h;
        if !inside(from) || !inside(to) {
            return false;
        }
        let (p0, p1) = (from.map(|v| v as i32), to.map(|v| v as i32));
        if p0 == p1 {
            let i = self.index(p0[0], p0[1]).expect("inside");
            return self.set(i, color);
        }
        let (dx, dy) = (to[0] - from[0], to[1] - from[1]);
        let n = (4.0 * (dx * dx + dy * dy).sqrt()) as i32;
        let mut changed = false;
        let mut prev = None;
        for i in 0..n {
            // With n = 1 there is a single sample: the start.
            let t = if n > 1 { f64::from(i) / f64::from(n - 1) } else { 0.0 };
            let p = [(dx * t + from[0]) as i32, (dy * t + from[1]) as i32];
            if prev != Some(p) {
                if let Some(i) = self.index(p[0], p[1]) {
                    changed |= self.set(i, color);
                }
                prev = Some(p);
            }
        }
        changed
    }

    /// 4-connected flood fill of the seed's exact colour. Returns the filled
    /// pixels (breadth-first order) and the other colours adjacent to them (ascending ARGB), or
    /// `None` if the seed is outside or already `color`.
    fn fill(&mut self, x: i32, y: i32, color: [u8; 4], collect: bool) -> Option<(Vec<usize>, BTreeSet<u32>)> {
        let seed = self.index(x, y)?;
        let old = self.image[seed];
        if old == color {
            return None;
        }
        let (w, h) = (self.width as i32, self.height as i32);
        let mut region = vec![seed];
        let mut others = BTreeSet::new();
        self.set(seed, color);
        let mut head = 0;
        while head < region.len() {
            let p = region[head] as i32;
            head += 1;
            for k in 0..4 {
                let (nx, ny) = (p % w + FILL_DX[k], p / w + FILL_DY[k]);
                if nx < 0 || ny < 0 || nx >= w || ny >= h {
                    continue;
                }
                let q = (ny * w + nx) as usize;
                let c = self.image[q];
                if c == old {
                    self.set(q, color);
                    region.push(q);
                } else if collect && c != color {
                    others.insert(argb(c));
                }
            }
        }
        Some((region, others))
    }

    /// Each pixel of the clicked same-colour component takes the adjacent colour
    /// closest to its reference pixel, `Σ_RGBA ((c − s)/255)²`; ties keep the lower ARGB value
    /// (the running best is stored as f32). The region is marked with a colour
    /// not in the palette while filling; any unused colour gives the same result.
    fn zap(&mut self, x: i32, y: i32) -> bool {
        let Some(seed) = self.index(x, y) else { return false };
        let old = self.image[seed];
        let used: std::collections::HashSet<[u8; 4]> = self.image.iter().copied().collect();
        let marker = (0..=u32::MAX).map(|v| v.to_le_bytes()).find(|c| !used.contains(c)).expect("a free marker colour");
        let Some((region, others)) = self.fill(x, y, marker, true) else { return false };
        let others: Vec<[u8; 4]> =
            others.into_iter().map(|v| [(v >> 16) as u8, (v >> 8) as u8, v as u8, (v >> 24) as u8]).collect();
        // With no other colour around (one segment fills the image) there is nothing to take;
        // restore the region instead.
        if others.is_empty() {
            for &p in &region {
                Arc::make_mut(&mut self.image)[p] = old;
            }
            return false;
        }
        let dist = |c: [u8; 4], s: [u8; 4]| -> f64 {
            (0..4).map(|k| ((f64::from(c[k]) - f64::from(s[k])) / 255.0).powi(2)).sum()
        };
        for &p in &region {
            let s = self.reference[p];
            let mut best = others[0];
            let mut bd = dist(best, s) as f32;
            for &c in &others[1..] {
                let d = dist(c, s);
                if d < f64::from(bd) {
                    best = c;
                    bd = d as f32;
                }
            }
            Arc::make_mut(&mut self.image)[p] = best;
        }
        true
    }
}

/// Remove Background: shapes left out of the export. Each click is kept as the point it hit and
/// whether it removed or restored the shape there, so the choice survives a new trace: the shape
/// under each point is looked up again ([`Background::resolve`]).
#[derive(Clone, Debug, Default)]
pub struct Background {
    hidden: Vec<bool>,
    undo: Vec<([f64; 2], bool)>,
    redo: Vec<([f64; 2], bool)>,
    revision: u64,
}

impl Background {
    pub fn hidden(&self) -> &[bool] {
        &self.hidden
    }

    pub fn is_hidden(&self, s: usize) -> bool {
        self.hidden.get(s).copied().unwrap_or(false)
    }

    /// Changes whenever the hidden shapes may have.
    pub fn revision(&self) -> u64 {
        self.revision
    }

    /// Whether the history has entries (the save path forces cut-outs then).
    pub fn has_edits(&self) -> bool {
        !self.undo.is_empty()
    }

    /// Remove (`remove`) or restore the shape at `p`.
    pub fn apply(&mut self, p: [f64; 2], remove: bool) {
        self.undo.push((p, remove));
        self.redo.clear();
    }

    pub fn undo(&mut self) -> bool {
        let Some(op) = self.undo.pop() else { return false };
        self.redo.push(op);
        true
    }

    pub fn redo(&mut self) -> bool {
        let Some(op) = self.redo.pop() else { return false };
        self.undo.push(op);
        true
    }

    /// Nothing removed; the redo list becomes the history in order.
    pub fn revert(&mut self) {
        self.redo.extend(std::mem::take(&mut self.undo).into_iter().rev());
    }

    /// Replay the history on a result with `segments` shapes; `shape_at` finds the shape under a
    /// point (points that miss every shape are skipped).
    pub fn resolve(&mut self, segments: usize, shape_at: impl Fn([f64; 2]) -> Option<usize>) {
        self.hidden = vec![false; segments];
        for &(p, remove) in &self.undo {
            if let Some(s) = shape_at(p).filter(|&s| s < segments) {
                self.hidden[s] = remove;
            }
        }
        self.revision += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use vectorizeo_core::Rgba;
    use vectorizeo_core::stages::segmenter::segmentation_from_colors;

    const W: [u8; 4] = [255, 255, 255, 255];
    const K: [u8; 4] = [0, 0, 0, 255];
    const G: [u8; 4] = [200, 200, 200, 255];

    fn editor(rows: &[&str], reference: &[[u8; 4]]) -> SegEditor {
        let (w, h) = (rows[0].len(), rows.len());
        let colors: Vec<[u8; 4]> = rows
            .iter()
            .flat_map(|r| {
                r.bytes().map(|b| match b {
                    b'w' => W,
                    b'k' => K,
                    _ => G,
                })
            })
            .collect();
        let img = RgbaImage::new(w, h, vec![Rgba { r: 0, g: 0, b: 0, a: 255 }; w * h]);
        let seg = segmentation_from_colors(&img, &colors);
        SegEditor::new(&seg, reference.to_vec())
    }

    #[test]
    fn fill_is_four_connected_and_undoable() {
        let mut e = editor(&["kw", "wk"], &[W; 4]);
        assert!(e.apply(Action::Fill { x: 0, y: 0, color: G }));
        assert_eq!(**e.image(), vec![G, W, W, K]);
        assert!(!e.apply(Action::Fill { x: 0, y: 0, color: G }), "same colour is no edit");
        e.undo();
        assert_eq!(**e.image(), vec![K, W, W, K]);
        assert!(!e.has_edits() && e.can_redo());
        e.redo();
        assert_eq!(**e.image(), vec![G, W, W, K]);
    }

    #[test]
    fn pencil_draws_a_line() {
        let mut e = editor(&["wwww"], &[W; 4]);
        assert!(e.apply(Action::Pencil { from: [0.5, 0.5], to: [3.5, 0.5], color: K }));
        assert_eq!(**e.image(), vec![K; 4]);
        assert!(!e.apply(Action::Pencil { from: [0.5, 0.5], to: [4.5, 0.5], color: W }), "end outside");
    }

    /// Each pixel of the zapped grey line goes to the neighbour colour closest to its input.
    #[test]
    fn zap_picks_nearest_neighbour_colour_per_pixel() {
        let reference = [W, W, W, [230; 4], [20, 20, 20, 255], K, K, K, K];
        let mut e = editor(&["www", "ggg", "kkk"], &reference);
        assert!(e.apply(Action::Zap { x: 1, y: 1 }));
        assert_eq!(&e.image()[3..6], &[W, K, K]);
    }

    #[test]
    fn revert_then_redo_replays_in_order() {
        let mut e = editor(&["ww"], &[W; 2]);
        e.apply(Action::Fill { x: 0, y: 0, color: K });
        e.apply(Action::Fill { x: 0, y: 0, color: G });
        e.undo();
        e.revert();
        assert_eq!(**e.image(), vec![W, W]);
        assert!(!e.can_undo());
        e.redo();
        assert_eq!(**e.image(), vec![K, K]);
        e.redo();
        assert_eq!(**e.image(), vec![G, G]);
    }

    /// Points pick the shapes on every replay; restoring and undo/redo/revert follow the history.
    #[test]
    fn background_history_replays_on_points() {
        let mut b = Background::default();
        // Shapes: x < 1 is shape 0, x < 2 shape 1, else shape 2.
        let at = |p: [f64; 2]| Some(p[0] as usize);
        b.apply([1.5, 0.0], true);
        b.apply([2.5, 0.0], true);
        b.apply([1.2, 0.0], false);
        b.resolve(3, at);
        assert_eq!(b.hidden(), &[false, false, true]);
        assert!(b.undo());
        b.resolve(3, at);
        assert_eq!(b.hidden(), &[false, true, true]);
        b.revert();
        b.resolve(3, at);
        assert_eq!(b.hidden(), &[false, false, false]);
        assert!(b.redo() && b.redo());
        // Another trace: the same points land on other shapes.
        b.resolve(2, |p| Some((p[0] as usize).min(1)));
        assert_eq!(b.hidden(), &[false, true]);
    }
}
