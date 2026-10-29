//! Pipeline stages, in the order they run:
//! classification → palette search → segmentation → contour smoothing → Bézier fitting.
mod ann;
pub mod bezier;
pub mod boundary;
pub mod classify;
mod coverage;
mod inversion;
mod labels;
pub mod palette;
pub mod preprocess;
pub mod segmenter;
mod shadow;
pub mod smoother;
pub mod subpixel;
