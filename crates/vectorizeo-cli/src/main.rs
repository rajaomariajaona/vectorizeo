use std::io;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use clap::builder::FalseyValueParser;
use clap::{Args, CommandFactory, Parser, Subcommand, ValueEnum, ValueHint};
use clap_complete::Shell;
use vectorizeo_core::RgbaImage;
use vectorizeo_core::export::{ExportOptions, Format, ShapeMode, export};
use vectorizeo_core::params::{EngineConfig, ImageType, Quality};
use vectorizeo_core::stages::classify::{PaletteMode, classify, recommend_from_palettes};
use vectorizeo_core::stages::palette::{cheapest, find_palettes, find_palettes_with_peaks};
use vectorizeo_core::stages::preprocess::preprocess;
use vectorizeo_core::stages::segmenter::Segmentation;

use vectorizeo_pipeline::{
    Flatten, extension, hex_rgba, load, palette_input, palette_params, parse_colors, segment_with, segmentation_image,
    vectorize,
};

mod interactive;
mod term;

/// Bitmap-to-vector tracer.
#[derive(Parser)]
#[command(name = "vectorizeo", version)]
struct Cli {
    #[command(subcommand)]
    cmd: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Segment the image and write each pixel in its segment's mean colour.
    Segment(Input),
    /// Run the full pipeline and write an SVG (EPS when the output ends in `.eps`).
    Trace {
        #[command(flatten)]
        input: Input,
        /// Skip Bezier fitting and write the smoothed polygons.
        #[arg(long, env = "VECTORIZEO_POLYGONS", value_parser = FalseyValueParser::new())]
        polygons: bool,
        /// Skip contour smoothing and write the raw pixel boundaries.
        #[arg(long, env = "VECTORIZEO_NO_SMOOTH", value_parser = FalseyValueParser::new())]
        no_smooth: bool,
        /// How shapes enclosed by other shapes are written.
        #[arg(short = 'm', long, value_enum, default_value_t = ShapeModeArg::Grouped)]
        shape_mode: ShapeModeArg,
        /// Stroke shape boundaries with a hairline in the fill colour.
        #[arg(long)]
        stroke: bool,
    },
    /// Propose palettes of `min`..=`max` colours and print them with their costs.
    Palette {
        /// Input bitmap.
        #[arg(value_hint = ValueHint::FilePath)]
        input: PathBuf,
        /// Image type (blended artwork weights edges down); `auto` classifies the image.
        #[arg(short = 't', long = "type", value_enum, default_value_t = TypeArg::Auto)]
        image_type: TypeArg,
        #[arg(short, long, value_name = "COLOR", default_value = "white", value_parser = parse_flatten)]
        flatten: Flatten,
        #[arg(long, default_value_t = 2)]
        min: usize,
        #[arg(long, default_value_t = 12)]
        max: usize,
    },
    /// Step through the GUI's settings interactively, with previews shown in the terminal.
    #[command(visible_alias = "i")]
    Interactive {
        /// Input bitmap (PNG, JPEG, BMP, GIF, TIFF).
        #[arg(value_hint = ValueHint::FilePath)]
        input: PathBuf,
        /// Output SVG (default: the input path with `.svg`).
        #[arg(value_hint = ValueHint::FilePath)]
        output: Option<PathBuf>,
    },
    /// Print a shell completion script to stdout.
    Completions { shell: Shell },
}

#[derive(Args)]
struct Input {
    /// Input bitmap (PNG, JPEG, BMP, GIF, TIFF).
    #[arg(value_hint = ValueHint::FilePath)]
    input: PathBuf,
    /// Output file.
    #[arg(value_hint = ValueHint::FilePath)]
    output: PathBuf,
    /// Image type preset; `auto` classifies the image as the GUI does on load.
    #[arg(short = 't', long = "type", value_enum, default_value_t = TypeArg::Auto)]
    image_type: TypeArg,
    /// Detail level preset; `auto` takes the classifier's level.
    #[arg(short, long, value_enum, default_value_t = QualityArg::Auto)]
    quality: QualityArg,
    /// Composite transparent pixels over a background: `white`, a colour `RRGGBB`, or `none` to
    /// keep the transparency.
    #[arg(short, long, value_name = "COLOR", default_value = "white", value_parser = parse_flatten)]
    flatten: Flatten,
    /// Fixed palette: a colour count (proposed by the palette finder), `auto` (the colour count the
    /// GUI recommends; may keep the flexible palette), or comma-separated `RRGGBB[AA]` colours.
    #[arg(short, long, value_name = "N|auto|COLORS", value_parser = parse_palette)]
    colors: Option<PaletteArg>,
}

#[derive(Clone)]
enum PaletteArg {
    Count(usize),
    Auto,
    List(Vec<[u8; 4]>),
}

fn parse_palette(s: &str) -> Result<PaletteArg, String> {
    if s == "auto" {
        return Ok(PaletteArg::Auto);
    }
    if let Ok(n) = s.parse::<usize>() {
        return if n >= 1 { Ok(PaletteArg::Count(n)) } else { Err("need at least one colour".into()) };
    }
    parse_colors(s).map(PaletteArg::List)
}

fn parse_flatten(s: &str) -> Result<Flatten, String> {
    s.parse()
}

#[derive(Clone, Copy, ValueEnum)]
enum TypeArg {
    /// Classify the image.
    Auto,
    Photo,
    /// Anti-aliased artwork.
    Blended,
    /// Artwork without anti-aliasing.
    Unblended,
}

#[derive(Clone, Copy, ValueEnum)]
enum ShapeModeArg {
    /// Stack shapes on top of each other.
    Stacked,
    /// Place shapes in cut-outs in the shapes below.
    Cutouts,
    /// Place shapes in cut-outs and group them by colour.
    Grouped,
}

#[derive(Clone, Copy, ValueEnum)]
enum QualityArg {
    /// Use the classifier's detail level.
    Auto,
    High,
    Medium,
    Low,
}

fn load_flattened(path: &Path, flatten: Flatten) -> Result<RgbaImage, String> {
    Ok(vectorizeo_pipeline::flatten(&load(path)?, flatten))
}

/// The palette that the palette search recommends after a 2..=12 run: the cheapest one when the colour-count
/// tree picks the flexible mode, else the tree's colour count. `None` for the flexible mode.
fn recommended_palette(img: &RgbaImage, image_type: ImageType) -> Option<Vec<[u8; 4]>> {
    let img = &palette_input(img, image_type);
    let (mut pals, peaks) = find_palettes_with_peaks(img, &palette_params(image_type), 2, 12);
    let rec = recommend_from_palettes(img, &pals, &peaks);
    eprintln!("colour count: {} ({:?})", rec.label, rec.mode);
    (rec.mode != PaletteMode::Flexible).then(|| pals.swap_remove(rec.colors - 2).colors)
}

/// Resolve `auto` type / quality with the classifier.
fn resolve_preset(img: &RgbaImage, path: &Path, t: TypeArg, q: QualityArg) -> (ImageType, Quality) {
    let class = matches!(t, TypeArg::Auto) || matches!(q, QualityArg::Auto);
    let c = class.then(|| {
        let c = classify(img, &extension(path));
        eprintln!("classified: {} ({:?}, {:?})", c.label, c.image_type, c.quality);
        c
    });
    let image_type = match t {
        TypeArg::Auto => c.as_ref().map(|c| c.image_type).unwrap(),
        TypeArg::Photo => ImageType::Photo,
        TypeArg::Unblended => ImageType::ArtworkUnblended,
        TypeArg::Blended => ImageType::ArtworkBlended,
    };
    let quality = match q {
        QualityArg::Auto => c.as_ref().map(|c| c.quality).unwrap(),
        QualityArg::High => Quality::High,
        QualityArg::Medium => Quality::Medium,
        QualityArg::Low => Quality::Low,
    };
    (image_type, quality)
}

/// The palette search: palettes for `min..=max` colours. For a 2..=12 run the GUI's
/// recommendation is marked, otherwise the cheapest palette.
fn palette(input: &Path, image_type: TypeArg, flatten: Flatten, min: usize, max: usize) -> Result<(), String> {
    if min == 0 {
        return Err("--min must be at least 1".into());
    }
    let img = load_flattened(input, flatten)?;
    let (image_type, _) = resolve_preset(&img, input, image_type, QualityArg::Medium);
    let img = palette_input(&img, image_type);
    let (pals, peaks) = find_palettes_with_peaks(&img, &palette_params(image_type), min, max);
    let mut best = cheapest(&pals);
    if (min, max) == (2, 12) {
        let rec = recommend_from_palettes(&img, &pals, &peaks);
        println!("colour count: {} ({:?})", rec.label, rec.mode);
        if rec.mode != PaletteMode::Flexible {
            best = Some(rec.colors - 2);
        }
    }
    for (i, p) in pals.iter().enumerate() {
        let colors: Vec<String> = p.colors.iter().map(|c| hex_rgba(*c)).collect();
        let mark = if Some(i) == best { "*" } else { " " };
        println!("{mark}{:>3}  {:.4}  {}", min + i, p.cost, colors.join(" "));
    }
    Ok(())
}

fn run_segmentation(input: &Input) -> Result<(Segmentation, EngineConfig, RgbaImage), String> {
    let img = load_flattened(&input.input, input.flatten)?;
    let (image_type, quality) = resolve_preset(&img, &input.input, input.image_type, input.quality);
    let cfg = EngineConfig::preset(image_type, quality);
    // The palette finder sees the flattened image; segmentation and smoothing the preprocessed one.
    let work = preprocess(&img, &cfg);
    let palette = match &input.colors {
        None => None,
        Some(PaletteArg::List(colors)) => Some(colors.clone()),
        Some(PaletteArg::Count(n)) => Some(
            find_palettes(&palette_input(&img, image_type), &palette_params(image_type), *n, *n).swap_remove(0).colors,
        ),
        Some(PaletteArg::Auto) => recommended_palette(&img, image_type),
    };
    if let Some(palette) = &palette {
        let hex: Vec<String> = palette.iter().map(|c| hex_rgba(*c)).collect();
        eprintln!("palette: {}", hex.join(" "));
    }
    let seg = segment_with(&work, &cfg, palette.as_deref());
    eprintln!("{} segments", seg.segments.len());
    Ok((seg, cfg, work))
}

fn segment(input: &Input) -> Result<(), String> {
    let (seg, _, _) = run_segmentation(input)?;
    segmentation_image(&seg).save(&input.output).map_err(|e| format!("cannot write {}: {e}", input.output.display()))
}

/// Output format from the file extension: `.eps` writes EPS, anything else SVG.
fn output_format(path: &std::path::Path) -> Format {
    Format::from_extension(&path.extension().unwrap_or_default().to_string_lossy())
}

/// Segment, trace, smooth, fit Beziers and write SVG (or EPS for a `.eps` output).
fn trace(input: &Input, polygons: bool, no_smooth: bool, opts: ExportOptions) -> Result<(), String> {
    let (seg, cfg, work) = run_segmentation(input)?;
    let v = vectorize(&seg, &work, &cfg, !no_smooth, !polygons);
    eprintln!("{} vertices", v.graph.vertices.len());
    let doc = export(&v.graph, v.curves.as_ref(), &v.colors, opts, &[], output_format(&input.output));
    std::fs::write(&input.output, doc).map_err(|e| format!("cannot write {}: {e}", input.output.display()))
}

fn main() -> ExitCode {
    let result = match Cli::parse().cmd {
        Command::Segment(input) => segment(&input),
        Command::Trace { input, polygons, no_smooth, shape_mode, stroke } => {
            let mode = match shape_mode {
                ShapeModeArg::Stacked => ShapeMode::Stacked,
                ShapeModeArg::Cutouts => ShapeMode::CutOuts,
                ShapeModeArg::Grouped => ShapeMode::CutOutsGrouped,
            };
            trace(&input, polygons, no_smooth, ExportOptions { mode, stroke })
        }
        Command::Palette { input, image_type, flatten, min, max } => palette(&input, image_type, flatten, min, max),
        Command::Interactive { input, output } => interactive::run(&input, output),
        Command::Completions { shell } => {
            clap_complete::generate(shell, &mut Cli::command(), "vectorizeo", &mut io::stdout());
            Ok(())
        }
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("{e}");
            ExitCode::FAILURE
        }
    }
}
