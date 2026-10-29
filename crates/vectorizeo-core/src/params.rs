//! Engine configuration.
//!
//! The engine is driven by text configs ("# Engine configuration") embedded in the crate and
//! parsed here. The nine presets are indexed `image_type * 3 + quality`.
//! User-facing advanced settings are mapped onto a config (see [`AdvancedSettings`]).

use std::fmt;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum ImageType {
    Photo = 0,
    /// "Artwork with blending" — anti-aliased edges.
    ArtworkBlended = 1,
    /// "Artwork without blending" — aliased, pixelated edges.
    ArtworkUnblended = 2,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum Quality {
    High = 0,
    Medium = 1,
    Low = 2,
}

const PRESETS: [&str; 9] = [
    include_str!("../presets/preset_0_photo_high.cfg"),
    include_str!("../presets/preset_1_photo_medium.cfg"),
    include_str!("../presets/preset_2_photo_low.cfg"),
    include_str!("../presets/preset_3_aa_high.cfg"),
    include_str!("../presets/preset_4_aa_medium.cfg"),
    include_str!("../presets/preset_5_aa_low.cfg"),
    include_str!("../presets/preset_6_aliased_high.cfg"),
    include_str!("../presets/preset_7_aliased_medium.cfg"),
    include_str!("../presets/preset_8_aliased_low.cfg"),
];
/// Base config for advanced mode (identical text to preset 4).
const ADVANCED_BASE: &str = include_str!("../presets/advanced_base.cfg");
/// Short pre-pass config (image_type_code 9, 5 segmenter iterations).
const PASS1: &str = include_str!("../presets/pass1.cfg");

/// Conjugate-gradient optimizer settings, one block per contour-smoother phase.
#[derive(Clone, Debug, Default, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct CgSettings {
    pub line_search_type: i32,
    pub iter_for_restart: i32,
    pub min_iter: i32,
    pub max_iter: i32,
    pub knock_out_count_down: i32,
    pub use_eps_from_step: i32,
    pub quad_line_srch_max_iter: i32,
    pub min_eps: f64,
    pub max_eps: f64,
    pub step_fraction: f64,
    pub rel_tol: f64,
    pub abs_tol: f64,
    pub quad_line_srch_rel_tol: f64,
    pub max_step_size: f64,
    pub num_blind_steps: i32,
    pub num_seeing_steps: i32,
    pub blind_blend_latest: f64,
}

/// All engine tunables. The field names are the keys of the text configs.
#[derive(Clone, Debug, Default, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct EngineConfig {
    // Shared
    pub is_anti_aliased: i32,
    pub image_type_code: i32,
    // Preprocessor
    pub filter_type: i32,
    pub reduce_noise_order: i32,
    // Palette finder et al.
    pub cost_per_color: f32,
    pub use_16_hist_bins: i32,
    pub use_26_connected_peaks: i32,
    pub edge_weight_scale: f64,
    pub edge_weight_floor: f64,
    // Segmenter
    pub seg_max_iterations: i32,
    pub lambda_pre: f32,
    pub lambda_initial: f32,
    pub lambda_final: f32,
    pub min_num_pixels: i32,
    pub num_iter_btwn_renum: i32,
    pub rise_fraction: f32,
    pub size_penalty_weight: f32,
    pub size_penalty_power: f32,
    pub use_over_seg_diag_extr: i32,
    // SubPixelSegmenter
    pub subpix_lambda_f: f32,
    pub subpix_beach_lambda_f: f32,
    pub subpix_color_cluster_margin: f32,
    pub subpix_use_connected_24: i32,
    // SuperPixelSegmenter
    pub superpix_lambda_f: f32,
    pub superpix_use_connected_24: i32,
    // SubPixelSegmentation
    pub subpixseg_self_weight_eps: f32,
    pub subpixseg_self_weight_penalty_low: f32,
    pub subpixseg_self_weight_penalty_high: f32,
    // SuperPixels
    pub superpixels_self_weight_eps: f32,
    pub superpixels_self_weight_penalty_low: f32,
    pub superpixels_self_weight_penalty_high: f32,
    // FlowField
    pub affinity_cost_scale: f32,
    // ContourSmoother (per-phase arrays)
    pub measurement_types: [i32; 3],
    pub optimizer_type: [i32; 3],
    pub prior_types: [i32; 3],
    pub prior_strengths: [f64; 3],
    pub length_penalty_weights: [f64; 3],
    pub do_puncture_corners: [i32; 3],
    pub smoother_enabled: [i32; 3],
    pub length_barrier_weight: f64,
    pub air_pressure_weight: f64,
    pub perturbation_range: f64,
    pub puncture_flatness_thresh: f64,
    pub puncture_corner_thresh: f64,
    pub anti_inv_pot_meas_scale: f64,
    pub anti_inv_pot_prior_scale: f64,
    pub phases: [CgSettings; 3],
    // BezierFitter
    pub bezier_max_iterations: i32,
    pub stat_thresh_initial: f64,
    pub stat_thresh_final: f64,
    // Fields set by the UI mapping but absent from the text format.
    pub flag_0x214: i32,
    pub cluster_colors: bool,
}

#[derive(Debug, PartialEq)]
pub enum ConfigError {
    UnknownKey(String),
    BadValue { key: String, value: String },
    WrongArity { key: String, expected: usize, got: usize },
}

impl fmt::Display for ConfigError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ConfigError::UnknownKey(k) => write!(f, "unknown config key `{k}`"),
            ConfigError::BadValue { key, value } => write!(f, "bad value `{value}` for `{key}`"),
            ConfigError::WrongArity { key, expected, got } => {
                write!(f, "`{key}` expects {expected} values, got {got}")
            }
        }
    }
}

impl std::error::Error for ConfigError {}

fn num<T: std::str::FromStr>(key: &str, v: &str) -> Result<T, ConfigError> {
    // Exponents may be zero-padded (`1e-005`); Rust's parser accepts the extra zeros.
    v.parse().map_err(|_| ConfigError::BadValue { key: key.into(), value: v.into() })
}

fn one<T: std::str::FromStr>(key: &str, vals: &[&str]) -> Result<T, ConfigError> {
    match vals {
        [v] => num(key, v),
        _ => Err(ConfigError::WrongArity { key: key.into(), expected: 1, got: vals.len() }),
    }
}

fn three<T: std::str::FromStr + Copy + Default>(key: &str, vals: &[&str]) -> Result<[T; 3], ConfigError> {
    if vals.len() != 3 {
        return Err(ConfigError::WrongArity { key: key.into(), expected: 3, got: vals.len() });
    }
    let mut out = [T::default(); 3];
    for (o, v) in out.iter_mut().zip(vals) {
        *o = num(key, v)?;
    }
    Ok(out)
}

impl EngineConfig {
    pub fn preset(image_type: ImageType, quality: Quality) -> Self {
        Self::parse(PRESETS[image_type as usize * 3 + quality as usize]).expect("embedded preset is valid")
    }

    pub fn advanced_base() -> Self {
        Self::parse(ADVANCED_BASE).expect("embedded preset is valid")
    }

    pub fn pass1() -> Self {
        Self::parse(PASS1).expect("embedded preset is valid")
    }

    /// Parse the "Name ....... value [value value]" text format.
    pub fn parse(text: &str) -> Result<Self, ConfigError> {
        let mut c = EngineConfig::default();
        for line in text.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let Some((key, rest)) = line.split_once(' ') else {
                continue;
            };
            let vals: Vec<&str> = rest.trim_start_matches([' ', '.']).split_whitespace().collect();
            c.set(key, &vals)?;
        }
        Ok(c)
    }

    fn set(&mut self, key: &str, v: &[&str]) -> Result<(), ConfigError> {
        if let Some(rest) = key.strip_prefix("ContourSmoother::phase_") {
            let (idx, field) = rest.split_once('.').ok_or_else(|| ConfigError::UnknownKey(key.into()))?;
            let i: usize = num(key, idx)?;
            let p = self.phases.get_mut(i).ok_or_else(|| ConfigError::UnknownKey(key.into()))?;
            match field {
                "cg_line_search_type" => p.line_search_type = one(key, v)?,
                "cg_iter_for_restart" => p.iter_for_restart = one(key, v)?,
                "cg_min_iter" => p.min_iter = one(key, v)?,
                "cg_max_iter" => p.max_iter = one(key, v)?,
                "cg_knock_out_count_down" => p.knock_out_count_down = one(key, v)?,
                "cg_use_eps_from_step" => p.use_eps_from_step = one(key, v)?,
                "cg_quad_line_srch_max_iter" => p.quad_line_srch_max_iter = one(key, v)?,
                "cg_min_eps" => p.min_eps = one(key, v)?,
                "cg_max_eps" => p.max_eps = one(key, v)?,
                "cg_step_fraction" => p.step_fraction = one(key, v)?,
                "cg_rel_tol" => p.rel_tol = one(key, v)?,
                "cg_abs_tol" => p.abs_tol = one(key, v)?,
                "cg_quad_line_srch_rel_tol" => p.quad_line_srch_rel_tol = one(key, v)?,
                "cg_max_step_size" => p.max_step_size = one(key, v)?,
                "cg_num_blind_steps" => p.num_blind_steps = one(key, v)?,
                "cg_num_seeing_steps" => p.num_seeing_steps = one(key, v)?,
                "cg_blind_blend_latest" => p.blind_blend_latest = one(key, v)?,
                _ => return Err(ConfigError::UnknownKey(key.into())),
            }
            return Ok(());
        }
        match key {
            "Shared::is_anti_aliased" => self.is_anti_aliased = one(key, v)?,
            "Shared::image_type_code" => self.image_type_code = one(key, v)?,
            "Preprocessor::filter_type" => self.filter_type = one(key, v)?,
            "Preprocessor::reduce_noise_order" => self.reduce_noise_order = one(key, v)?,
            "PaletteFinder::cost_per_color" => self.cost_per_color = one(key, v)?,
            "PaletteFinder::use_16_hist_bins" => self.use_16_hist_bins = one(key, v)?,
            "PaletteFinder::use_26_connected_peaks" => self.use_26_connected_peaks = one(key, v)?,
            "EdgeImage::edge_weight_scale" => self.edge_weight_scale = one(key, v)?,
            "EdgeImage::edge_weight_floor" => self.edge_weight_floor = one(key, v)?,
            "Segmenter::max_iterations" => self.seg_max_iterations = one(key, v)?,
            "Segmenter::lambda_pre" => self.lambda_pre = one(key, v)?,
            "Segmenter::lambda_initial" => self.lambda_initial = one(key, v)?,
            "Segmenter::lambda_final" => self.lambda_final = one(key, v)?,
            "Segmenter::min_num_pixels" => self.min_num_pixels = one(key, v)?,
            "Segmenter::num_iter_btwn_renum" => self.num_iter_btwn_renum = one(key, v)?,
            "Segmenter::rise_fraction" => self.rise_fraction = one(key, v)?,
            "Segmenter::size_penalty_weight" => self.size_penalty_weight = one(key, v)?,
            "Segmenter::size_penalty_power" => self.size_penalty_power = one(key, v)?,
            "Segmenter::use_over_seg_diag_extr" => self.use_over_seg_diag_extr = one(key, v)?,
            "SubPixelSegmenter::lambda_f" => self.subpix_lambda_f = one(key, v)?,
            "SubPixelSegmenter::beach_lambda_f" => self.subpix_beach_lambda_f = one(key, v)?,
            "SubPixelSegmenter::color_cluster_margin" => self.subpix_color_cluster_margin = one(key, v)?,
            "SubPixelSegmenter::use_connected_24_iteration" => self.subpix_use_connected_24 = one(key, v)?,
            "SuperPixelSegmenter::lambda_f" => self.superpix_lambda_f = one(key, v)?,
            "SuperPixelSegmenter::use_connected_24_iteration" => self.superpix_use_connected_24 = one(key, v)?,
            "SubPixelSegmentation::self_weight_eps" => self.subpixseg_self_weight_eps = one(key, v)?,
            "SubPixelSegmentation::self_weight_penalty_low" => self.subpixseg_self_weight_penalty_low = one(key, v)?,
            "SubPixelSegmentation::self_weight_penalty_high" => self.subpixseg_self_weight_penalty_high = one(key, v)?,
            "SuperPixels::self_weight_eps" => self.superpixels_self_weight_eps = one(key, v)?,
            "SuperPixels::self_weight_penalty_low" => self.superpixels_self_weight_penalty_low = one(key, v)?,
            "SuperPixels::self_weight_penalty_high" => self.superpixels_self_weight_penalty_high = one(key, v)?,
            "FlowField::affinity_cost_scale" => self.affinity_cost_scale = one(key, v)?,
            "ContourSmoother::measurement_types" => self.measurement_types = three(key, v)?,
            "ContourSmoother::optimizer_type" => self.optimizer_type = three(key, v)?,
            "ContourSmoother::prior_types" => self.prior_types = three(key, v)?,
            "ContourSmoother::prior_strengths" => self.prior_strengths = three(key, v)?,
            "ContourSmoother::length_penalty_weights" => self.length_penalty_weights = three(key, v)?,
            "ContourSmoother::do_puncture_corners" => self.do_puncture_corners = three(key, v)?,
            "ContourSmoother::is_enabled" => self.smoother_enabled = three(key, v)?,
            "ContourSmoother::length_barrier_weight" => self.length_barrier_weight = one(key, v)?,
            "ContourSmoother::air_pressure_weight" => self.air_pressure_weight = one(key, v)?,
            "ContourSmoother::perturbation_range" => self.perturbation_range = one(key, v)?,
            "ContourSmoother::puncture_flatness_thresh" => self.puncture_flatness_thresh = one(key, v)?,
            "ContourSmoother::puncture_corner_thresh" => self.puncture_corner_thresh = one(key, v)?,
            "ContourSmoother::anti_inv_pot_meas_scale" => self.anti_inv_pot_meas_scale = one(key, v)?,
            "ContourSmoother::anti_inv_pot_prior_scale" => self.anti_inv_pot_prior_scale = one(key, v)?,
            "BezierFitter::max_iterations" => self.bezier_max_iterations = one(key, v)?,
            "BezierFitter::stat_thresh_initial" => self.stat_thresh_initial = one(key, v)?,
            "BezierFitter::stat_thresh_final" => self.stat_thresh_final = one(key, v)?,
            _ => return Err(ConfigError::UnknownKey(key.into())),
        }
        Ok(())
    }
}

/// Anti-aliasing artifact rejection level (`p[7]` in).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AaRejection {
    Level0 = 0,
    Level1 = 1,
    Level2 = 2,
}

/// The advanced-mode UI settings (the `adv_*` keys) that maps onto an [`EngineConfig`].
#[derive(Clone, Debug, PartialEq)]
pub struct AdvancedSettings {
    /// `adv_segmentation_complexity`, 1..=12 (higher → more segments).
    pub segmentation_complexity: i32,
    /// `adv_min_num_pixels`.
    pub min_num_pixels: i32,
    /// `adv_anti_aliasing_rejection`.
    pub aa_rejection: AaRejection,
    /// `adv_cluster_colors`.
    pub cluster_colors: bool,
    /// `adv_contour_smoothness`, 1..=12.
    pub smoothness: i32,
    /// `adv_use_contour_aa` — the "anti-aliasing engine" checkbox.
    pub anti_aliasing_engine: bool,
    /// `adv_detect_sharp_corners`.
    pub detect_sharp_corners: bool,
    /// `adv_curve_complexity` (number of nodes), 1..=12.
    pub num_nodes: i32,
}

/// The default advanced settings.
impl Default for AdvancedSettings {
    fn default() -> Self {
        Self {
            segmentation_complexity: 5,
            min_num_pixels: 0,
            aa_rejection: AaRejection::Level0,
            cluster_colors: true,
            smoothness: 6,
            anti_aliasing_engine: true,
            detect_sharp_corners: true,
            num_nodes: 6,
        }
    }
}

/// `lo * (hi/lo)^(t/11)` .
fn geometric(lo: f64, hi: f64, t: i32) -> f64 {
    lo * (hi / lo).powf(f64::from(t) / 11.0)
}

impl EngineConfig {
    /// Segmentation-stage overrides.
    pub fn apply_advanced_segmentation(&mut self, s: &AdvancedSettings) {
        let c = s.segmentation_complexity;
        self.lambda_initial = ((1.0 + f64::from(12 - c)) * 0.0001) as f32;
        self.lambda_pre = self.lambda_initial;
        self.lambda_final = geometric(0.05, 0.05 * 400.0, 12 - c) as f32;
        self.min_num_pixels = s.min_num_pixels;
        let (flag, aa) = match s.aa_rejection {
            AaRejection::Level0 => (1, 1),
            AaRejection::Level1 => (0, 1),
            AaRejection::Level2 => (0, 0),
        };
        self.flag_0x214 = flag;
        self.is_anti_aliased = aa;
        self.cluster_colors = s.cluster_colors;
    }

    /// Smoothing/fitting-stage overrides (with `param_2` other than 1 or 2).
    pub fn apply_advanced_smoothing(&mut self, s: &AdvancedSettings) {
        const PRIOR_LO: [f64; 3] = [0.05, 0.01, 0.45]; // /
        let (prior_hi, len_lo, len_hi) = if s.anti_aliasing_engine {
            ([10.0; 3], [0.25, 0.05, 0.005], [10.0, 1.0, 1.0]) //
        } else {
            ([20.0; 3], [0.5; 3], [40.0; 3]) //
        };
        let aa = i32::from(s.anti_aliasing_engine);
        for i in 0..3 {
            self.measurement_types[i] = aa;
            self.prior_types[i] = 3;
            self.prior_strengths[i] = geometric(PRIOR_LO[i], prior_hi[i], s.smoothness - 1);
            self.length_penalty_weights[i] = geometric(len_lo[i], len_hi[i], s.smoothness - 1);
            self.do_puncture_corners[i] = 0;
            self.smoother_enabled[i] = 1;
        }
        if s.detect_sharp_corners {
            self.do_puncture_corners[1] = 1;
        } else {
            self.smoother_enabled[1] = 0;
        }
        let n = s.num_nodes;
        self.stat_thresh_initial = (1.0 + f64::from(12 - n)) * 0.0001;
        self.stat_thresh_final = geometric(0.1, 0.1 * 200.0, 12 - n);
        self.is_anti_aliased = aa;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_presets_parse() {
        for text in PRESETS.iter().chain([&ADVANCED_BASE, &PASS1]) {
            EngineConfig::parse(text).unwrap();
        }
    }

    #[test]
    fn preset_values() {
        let c = EngineConfig::preset(ImageType::ArtworkUnblended, Quality::Medium);
        assert_eq!(c.image_type_code, 1);
        assert_eq!(c.lambda_final, 0.5);
        assert_eq!(c.prior_strengths, [10.0, 10.0, 20.0]);
        assert_eq!(c.length_penalty_weights, [5.0, 0.5, 0.1]);
        assert_eq!(c.phases[2].max_iter, 500);
        assert_eq!(c.subpix_beach_lambda_f, 1e-5);
        assert_eq!(c.stat_thresh_final, 0.6);

        let p = EngineConfig::preset(ImageType::Photo, Quality::Low);
        assert_eq!(p.image_type_code, 6);
        assert_eq!(p.smoother_enabled, [1, 0, 0]);
        assert_eq!(p.stat_thresh_final, 10.0);
    }

    #[test]
    fn advanced_base_is_preset_4() {
        assert_eq!(EngineConfig::advanced_base(), EngineConfig::preset(ImageType::ArtworkBlended, Quality::Medium));
    }

    #[test]
    fn advanced_mapping_endpoints() {
        let mut s = AdvancedSettings {
            segmentation_complexity: 12,
            min_num_pixels: 3,
            aa_rejection: AaRejection::Level1,
            cluster_colors: true,
            smoothness: 1,
            anti_aliasing_engine: true,
            detect_sharp_corners: true,
            num_nodes: 12,
        };
        let mut c = EngineConfig::advanced_base();
        c.apply_advanced_segmentation(&s);
        c.apply_advanced_smoothing(&s);
        assert!((c.lambda_initial - 1e-4).abs() < 1e-9);
        assert!((c.lambda_final - 0.05).abs() < 1e-7);
        assert_eq!(c.prior_strengths, [0.05, 0.01, 0.45]);
        assert_eq!(c.do_puncture_corners, [0, 1, 0]);
        assert!((c.stat_thresh_final - 0.1).abs() < 1e-12);

        s.segmentation_complexity = 1;
        s.smoothness = 12;
        s.num_nodes = 1;
        c.apply_advanced_segmentation(&s);
        c.apply_advanced_smoothing(&s);
        assert!((c.lambda_final - 20.0).abs() < 1e-4);
        assert!((c.prior_strengths[2] - 10.0).abs() < 1e-9);
        assert!((c.stat_thresh_final - 20.0).abs() < 1e-9);
    }
}
