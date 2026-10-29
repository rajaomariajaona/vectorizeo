//! Background work (image loading through Bézier fitting, plus rasterising and
//! hover previews) as values: natively a thread runs them, on the web a Web Worker
//! (`src/bin/vectorizeo-worker.rs`) receives them postcard-encoded.

use std::path::PathBuf;
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use vectorizeo_core::RgbaImage;
use vectorizeo_core::stages::classify::Classification;
use vectorizeo_pipeline::session::{Done, Job, Stage, classify_loaded};
use vectorizeo_pipeline::{Flatten, extension, flatten, premultiplied, render_svg};

/// Straight-alpha RGBA pixels (`image::RgbaImage` has no serde support).
#[derive(Clone, Serialize, Deserialize)]
pub struct Image {
    pub width: u32,
    pub height: u32,
    pub rgba: Vec<u8>,
}

impl From<image::RgbaImage> for Image {
    fn from(img: image::RgbaImage) -> Self {
        Image { width: img.width(), height: img.height(), rgba: img.into_raw() }
    }
}

/// Output of loading and classifying an image.
#[derive(Serialize, Deserialize)]
pub struct Loaded {
    pub path: PathBuf,
    /// As decoded (straight alpha), for display.
    pub original: Image,
    pub loaded: RgbaImage,
    pub class: Classification,
}

#[derive(Serialize, Deserialize)]
pub enum Task {
    /// Decode and classify a file, or its contents when given
    /// (browser picks and drops).
    Load { path: PathBuf, bytes: Option<Vec<u8>> },
    /// A pipeline stage.
    Stage(Job),
    /// Rasterise an SVG at `scale` × its size.
    Render { svg: String, scale: f32 },
    /// The input flattened another way, or with every pixel rounded to the nearest colour of
    /// `palette` (the transparency hover preview and the palette "Compare").
    Preview { loaded: Arc<RgbaImage>, flatten: Flatten, palette: Option<Vec<[u8; 4]>> },
}

#[derive(Serialize, Deserialize)]
pub enum Output {
    Loaded(Result<Loaded, String>),
    Done(Done),
    Rendered(Option<Image>),
    Preview(Image),
    /// The web worker died (a panic); never produced by [`Task::run`].
    Failed(String),
}

/// What the web worker sends back.
#[derive(Serialize, Deserialize)]
pub enum Reply {
    /// The worker's wasm is running and takes tasks.
    Ready,
    Stage(String),
    Done(Output),
}

impl Task {
    /// Progress label while the task starts.
    pub fn label(&self) -> &'static str {
        match self {
            Task::Load { .. } => "Loading Image",
            Task::Stage(job) => job.first_stage().label(),
            Task::Render { .. } => "Rendering",
            Task::Preview { .. } => "Preview",
        }
    }

    /// Do the work; `stage` gets the label of each later stage.
    pub fn run(self, stage: &mut dyn FnMut(&str)) -> Output {
        match self {
            Task::Load { path, bytes } => Output::Loaded(load(path, bytes, stage)),
            Task::Stage(job) => Output::Done(job.run(&mut |s| stage(s.label()))),
            Task::Render { svg, scale } => Output::Rendered(render_svg(&svg, scale).map(Image::from)),
            Task::Preview { loaded, flatten, palette } => {
                Output::Preview(preview(&loaded, flatten, palette.as_deref()))
            }
        }
    }
}

fn load(path: PathBuf, bytes: Option<Vec<u8>>, stage: &mut dyn FnMut(&str)) -> Result<Loaded, String> {
    let decoded = match &bytes {
        Some(bytes) => image::load_from_memory(bytes),
        None => image::open(&path),
    };
    let original = decoded.map_err(|e| format!("Cannot open {}: {e}", path.display()))?.to_rgba8();
    let loaded = premultiplied(&original);
    stage(Stage::Classify.label());
    let class = classify_loaded(&loaded, &extension(&path));
    Ok(Loaded { path, original: original.into(), loaded, class })
}

/// The input flattened with `f`, straight alpha, optionally rounded to `palette`.
pub fn preview(loaded: &RgbaImage, f: Flatten, palette: Option<&[[u8; 4]]>) -> Image {
    let img = flatten(loaded, f);
    let straight = |p: &vectorizeo_core::Rgba| -> [u8; 4] {
        let a = u32::from(p.a);
        if a == 0 {
            return [0; 4];
        }
        let c = |v: u8| ((u32::from(v) * 255 + a / 2) / a).min(255) as u8;
        [c(p.r), c(p.g), c(p.b), p.a]
    };
    let mut rgba = Vec::with_capacity(img.pixels.len() * 4);
    for p in &img.pixels {
        let mut c = straight(p);
        if let Some(pal) = palette {
            let dist = |q: &[u8; 4]| (0..4).map(|i| (i32::from(q[i]) - i32::from(c[i])).pow(2)).sum::<i32>();
            c = pal.iter().copied().min_by_key(dist).unwrap_or(c);
        }
        rgba.extend_from_slice(&c);
    }
    Image { width: img.width as u32, height: img.height as u32, rgba }
}

#[cfg(test)]
mod tests {
    use super::*;
    use vectorizeo_core::Rgba;

    /// Palette previews round every pixel to the nearest colour; flatten previews composite.
    #[test]
    fn previews() {
        let px = |r, g, b, a| Rgba { r, g, b, a };
        let mut img = RgbaImage::new(2, 1, vec![px(250, 10, 10, 255), px(0, 0, 0, 0)]);
        img.premultiply();
        let pal = vec![[255, 0, 0, 255], [255, 255, 255, 255]];
        let out = preview(&img, Flatten::WHITE, Some(&pal));
        assert_eq!(out.rgba, [255, 0, 0, 255, 255, 255, 255, 255]);
        let out = preview(&img, Flatten(None), None);
        assert_eq!(out.rgba, [250, 10, 10, 255, 0, 0, 0, 0]);
    }

    /// Tasks and outputs survive the postcard round trip the web worker uses.
    #[test]
    fn preview_task_round_trip() {
        let img = RgbaImage::new(1, 1, vec![Rgba { r: 1, g: 2, b: 3, a: 255 }]);
        let task = Task::Preview { loaded: Arc::new(img), flatten: Flatten::WHITE, palette: None };
        let task: Task = postcard::from_bytes(&postcard::to_allocvec(&task).unwrap()).unwrap();
        let out = postcard::to_allocvec(&Reply::Done(task.run(&mut |_| {}))).unwrap();
        let Reply::Done(Output::Preview(img)) = postcard::from_bytes(&out).unwrap() else { panic!("not a preview") };
        assert_eq!(img.rgba, [1, 2, 3, 255]);
    }
}
