//! Pixel label map shared by the segmenters (labels plus union-find parents).

use crate::image::RgbaImage;

/// 1/255 as f32.
pub(crate) const INV_255_F32: f32 = 0.003921569;

/// Neighbour offsets. The first 8 entries are the
/// 8-neighbourhood; the 24-neighbourhood tables
/// repeat them and append the ring at distance 2.
pub(crate) const NBR_DX: [i32; 24] = [1, 0, -1, 0, -1, 1, -1, 1, -2, -2, -2, -2, -2, 2, 2, 2, 2, 2, -1, 0, 1, -1, 0, 1];
pub(crate) const NBR_DY: [i32; 24] = [0, 1, 0, -1, 1, -1, -1, 1, -2, -1, 0, 1, 2, -2, -1, 0, 1, 2, -2, -2, -2, 2, 2, 2];

/// The engine's LCG, `seed = seed·2003 mod 9973`, value `seed / 9973`. The seed
/// starts at 1 for every run.
pub(crate) struct Lcg(pub(crate) i32);

impl Lcg {
    pub(crate) fn next(&mut self) -> f64 {
        self.0 = self.0 * 0x7d3 % 0x26f5;
        f64::from(self.0) * 0.0001002707309736288
    }
}

/// Number of neighbour offsets used: 8, or 24 when the 24-connected mode is on.
pub(crate) fn nbr_count(conn24: bool) -> usize {
    if conn24 { 24 } else { 8 }
}

/// Root lookup with one-level path compression into both tables.
pub(crate) fn find(labels: &mut [i32], parent: &mut [i32], p: usize) -> i32 {
    let l = labels[p];
    let mut r = parent[l as usize];
    if r >= 0 {
        let mut root = l;
        while r >= 0 {
            root = r;
            r = parent[root as usize];
        }
        parent[l as usize] = root;
        labels[p] = root;
        return root;
    }
    l
}

/// `find` at (x, y), or -1 outside the image.
pub(crate) fn find_xy(labels: &mut [i32], parent: &mut [i32], w: i32, h: i32, x: i32, y: i32) -> i32 {
    if x < 0 || y < 0 || x >= w || y >= h {
        return -1;
    }
    find(labels, parent, (y * w + x) as usize)
}

/// Relabel 4-connected components of equal root in raster order.
/// Returns the new labels and the number of components.
pub(crate) fn relabel_components(w: i32, h: i32, roots: &[i32]) -> (Vec<i32>, usize) {
    let n = (w * h) as usize;
    let mut new = vec![-1i32; n];
    let mut count = 0i32;
    let mut stack = Vec::new();
    for start in 0..n {
        if new[start] >= 0 {
            continue;
        }
        let old = roots[start];
        new[start] = count;
        stack.push(start as i32);
        while let Some(p) = stack.pop() {
            let (x, y) = (p % w, p / w);
            for (dx, dy) in [(1, 0), (-1, 0), (0, 1), (0, -1)] {
                let (nx, ny) = (x + dx, y + dy);
                if nx < 0 || ny < 0 || nx >= w || ny >= h {
                    continue;
                }
                let q = (ny * w + nx) as usize;
                if new[q] < 0 && roots[q] == old {
                    new[q] = count;
                    stack.push(q as i32);
                }
            }
        }
        count += 1;
    }
    (new, count as usize)
}

fn rgba(img: &RgbaImage, x: i32, y: i32) -> [f32; 4] {
    let c = img.pixels[y as usize * img.width + x as usize];
    [c.r, c.g, c.b, c.a].map(f32::from)
}

/// Squared colour distance between two pixels, in 0..1 units.
fn pixel_dist2(img: &RgbaImage, x1: i32, y1: i32, x2: i32, y2: i32) -> f32 {
    let (a, b) = (rgba(img, x1, y1), rgba(img, x2, y2));
    let d = [0, 1, 2, 3].map(|i| (a[i] - b[i]) * INV_255_F32);
    (f64::from(d[0] * d[0]) + f64::from(d[1] * d[1]) + f64::from(d[2] * d[2]) + f64::from(d[3] * d[3])) as f32
}

/// Squared second difference `p1 − 2·p2 + p3` (0..1 units); 0 if an end is outside.
fn second_diff2(img: &RgbaImage, (x1, y1): (i32, i32), (x2, y2): (i32, i32), (x3, y3): (i32, i32)) -> f64 {
    let (w, h) = (img.width as i32, img.height as i32);
    if x1 < 0 || x1 >= w || y1 < 0 || y1 >= h || x3 < 0 || x3 >= w || y3 < 0 || y3 >= h {
        return 0.0;
    }
    let (a, b, c) = (rgba(img, x1, y1), rgba(img, x2, y2), rgba(img, x3, y3));
    (0..4)
        .map(|i| {
            let d = f64::from(((a[i] - b[i] * 2.0) + c[i]) * INV_255_F32);
            d * d
        })
        .sum()
}

/// True if (x,y), (x,y+dy) and (x+dx,y) share a label that (x+dx,y+dy) lacks,
/// i.e. moving (x,y) would break an L-shaped corner of its segment.
fn is_corner(labels: &mut [i32], parent: &mut [i32], w: i32, h: i32, x: i32, y: i32, dx: i32, dy: i32) -> bool {
    let a = find_xy(labels, parent, w, h, x, y);
    let b = find_xy(labels, parent, w, h, x, y + dy);
    let c = find_xy(labels, parent, w, h, x + dx, y);
    let d = find_xy(labels, parent, w, h, x + dx, y + dy);
    a == b && a == c && a != d
}

/// Resolve 2×2 blocks where segments touch only diagonally by relabelling one
/// pixel so the chosen diagonal becomes 4-connected. In a full checkerboard the diagonal with
/// the smaller colour curvature wins; ties leave the block unchanged.
pub(crate) fn fix_diagonals(img: &RgbaImage, labels: &mut [i32], parent: &mut [i32]) {
    let (w, h) = (img.width as i32, img.height as i32);
    for y in 0..h - 1 {
        for x in 0..w - 1 {
            let tl = find_xy(labels, parent, w, h, x, y);
            let tr = find_xy(labels, parent, w, h, x + 1, y);
            let bl = find_xy(labels, parent, w, h, x, y + 1);
            let br = find_xy(labels, parent, w, h, x + 1, y + 1);
            if tl == bl || tl == tr || br == bl || br == tr {
                continue;
            }
            let anti = bl == tr;
            // true: join TL–BR (main diagonal); false: join BL–TR.
            let main = if tl == br {
                if anti {
                    let (mut main_curv, mut anti_curv) = (0.0f32, 0.0f32);
                    for j in 0..2 {
                        for i in 0..2 {
                            let (cx, cy) = (x + i, y + j);
                            let m = second_diff2(img, (cx - 1, cy - 1), (cx, cy), (cx + 1, cy + 1));
                            main_curv = (m + f64::from(main_curv)) as f32;
                            let a = second_diff2(img, (cx - 1, cy + 1), (cx, cy), (cx + 1, cy - 1));
                            anti_curv = (a + f64::from(anti_curv)) as f32;
                        }
                    }
                    if anti_curv < main_curv {
                        false
                    } else if anti_curv <= main_curv {
                        continue;
                    } else {
                        true
                    }
                } else {
                    true
                }
            } else if anti {
                false
            } else {
                continue;
            };
            // Candidate pixels to relabel (p1, p2), the corner probe directions and the
            // reference pixels (r1 on the target diagonal, r2 = the other one).
            let (target, p1, p2, d1, r1, r2) = if main {
                (tl, (x + 1, y), (x, y + 1), (1, -1), (x + 1, y + 1), (x, y))
            } else {
                (bl, (x, y), (x + 1, y + 1), (-1, -1), (x + 1, y), (x, y + 1))
            };
            let c1 = is_corner(labels, parent, w, h, p1.0, p1.1, d1.0, d1.1);
            let c2 = is_corner(labels, parent, w, h, p2.0, p2.1, -d1.0, -d1.1);
            let pick = match (c1, c2) {
                (false, false) => {
                    let f12 = pixel_dist2(img, r1.0, r1.1, p1.0, p1.1);
                    let f13 = pixel_dist2(img, r2.0, r2.1, p1.0, p1.1);
                    let f14 = pixel_dist2(img, r1.0, r1.1, p2.0, p2.1);
                    let f15 = pixel_dist2(img, r2.0, r2.1, p2.0, p2.1);
                    if f64::from(f15) + f64::from(f14) <= f64::from(f13 + f12) { p2 } else { p1 }
                }
                (false, true) => p1,
                (true, false) => p2,
                (true, true) => continue,
            };
            labels[(pick.1 * w + pick.0) as usize] = target;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Rgba;

    #[test]
    fn diagonal_fix_joins_a_single_diagonal() {
        // Labels: A B / B' A where B and B' are different segments: only A touches diagonally.
        let px = |v: u8| Rgba { r: v, g: v, b: v, a: 255 };
        let img = RgbaImage::new(2, 2, vec![px(0), px(200), px(100), px(0)]);
        let mut labels = vec![0, 1, 2, 0];
        let mut parent = vec![-1; 3];
        fix_diagonals(&img, &mut labels, &mut parent);
        // One of the off-diagonal pixels joins segment 0, making it 4-connected.
        let zeros = labels.iter().filter(|&&l| l == 0).count();
        assert_eq!(zeros, 3, "{labels:?}");
    }
}
