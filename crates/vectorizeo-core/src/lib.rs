//! Bitmap vectorization engine: pixel segmentation, boundary tracing, contour smoothing and
//! Bézier fitting, with SVG and EPS writers.
//!
//! The modules follow the pipeline: [`params`] (configuration), [`stages`] (the processing
//! stages), [`export`], [`svg`] and [`eps`] (output).

pub mod eps;
pub mod export;
pub mod image;
pub mod params;
#[cfg(feature = "serde")]
mod serde_util;
pub mod stages;
pub mod svg;

pub use image::{Rgba, RgbaImage};
pub use params::{AdvancedSettings, EngineConfig, ImageType, Quality};
