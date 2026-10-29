//! Terminal helpers for the interactive wizard: inline images (kitty graphics protocol, or
//! half-block characters elsewhere), colour swatches and line-based prompts.

use std::io::{self, BufRead, Write};

use base64::Engine;
use base64::engine::general_purpose::STANDARD as BASE64;
use crossterm::style::Stylize;

/// Leaving the current prompt: one step back, or out of the wizard.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Nav {
    Back,
    Quit,
}

/// Terminals that implement the kitty graphics protocol.
fn has_kitty_graphics() -> bool {
    match std::env::var("VECTORIZEO_GRAPHICS").as_deref() {
        Ok("kitty") => return true,
        Ok("blocks") => return false,
        _ => {}
    }
    let term = std::env::var("TERM").unwrap_or_default();
    let program = std::env::var("TERM_PROGRAM").unwrap_or_default();
    term.contains("kitty")
        || term.contains("ghostty")
        || std::env::var_os("KITTY_WINDOW_ID").is_some()
        || matches!(program.as_str(), "WezTerm" | "ghostty")
}

/// Terminal size in cells and the size of one cell in pixels (8×16 when the terminal does not
/// report pixels).
fn geometry() -> (u32, u32, f64, f64) {
    let (cols, rows, w, h) = match crossterm::terminal::window_size() {
        Ok(s) => (u32::from(s.columns), u32::from(s.rows), u32::from(s.width), u32::from(s.height)),
        Err(_) => {
            crossterm::terminal::size().map(|(c, r)| (u32::from(c), u32::from(r), 0, 0)).unwrap_or((80, 24, 0, 0))
        }
    };
    let (cols, rows) = (cols.max(20), rows.max(10));
    let cell_w = if w > 0 { f64::from(w) / f64::from(cols) } else { 8.0 };
    let cell_h = if h > 0 { f64::from(h) / f64::from(rows) } else { 16.0 };
    (cols, rows, cell_w, cell_h)
}

/// Light/dark grey checkerboard behind transparent pixels, as the GUI shows them.
fn composite_checkerboard(img: &image::RgbaImage) -> image::RgbaImage {
    let mut out = img.clone();
    for (x, y, px) in out.enumerate_pixels_mut() {
        let a = u32::from(px[3]);
        if a == 255 {
            continue;
        }
        let bg: u32 = if (x / 8 + y / 8) % 2 == 0 { 0xff } else { 0xcc };
        for c in 0..3 {
            px[c] = ((u32::from(px[c]) * a + bg * (255 - a) + 127) / 255) as u8;
        }
        px[3] = 255;
    }
    out
}

/// Print an image (straight RGBA) at up to twice its size, fitted into the terminal width and
/// most of its height, then move to the next line.
pub fn show_image(img: &image::RgbaImage) {
    let (cols, rows, cell_w, cell_h) = geometry();
    let (w, h) = (f64::from(img.width().max(1)), f64::from(img.height().max(1)));
    let max_w = f64::from(cols - 2) * cell_w;
    let max_h = f64::from(rows.saturating_sub(8).max(6)) * cell_h;
    let scale = (max_w / w).min(max_h / h).min(2.0);
    let out_cols = ((w * scale / cell_w).round() as u32).clamp(1, cols - 2);
    let out_rows = ((h * scale / cell_h).round() as u32).max(1);
    let img = composite_checkerboard(img);
    let mut out = io::stdout().lock();
    if has_kitty_graphics() {
        let mut png = Vec::new();
        if image::DynamicImage::ImageRgba8(img)
            .write_to(&mut io::Cursor::new(&mut png), image::ImageFormat::Png)
            .is_err()
        {
            return;
        }
        let data = BASE64.encode(png);
        let chunks: Vec<&[u8]> = data.as_bytes().chunks(4096).collect();
        for (i, chunk) in chunks.iter().enumerate() {
            let more = u8::from(i + 1 < chunks.len());
            let chunk = std::str::from_utf8(chunk).unwrap_or_default();
            if i == 0 {
                let _ = write!(out, "\x1b_Ga=T,f=100,q=2,c={out_cols},r={out_rows},m={more};{chunk}\x1b\\");
            } else {
                let _ = write!(out, "\x1b_Gm={more};{chunk}\x1b\\");
            }
        }
        let _ = writeln!(out);
    } else {
        // Two image rows per character cell: foreground = upper half, background = lower half.
        let small = image::imageops::resize(&img, out_cols, out_rows * 2, image::imageops::FilterType::Triangle);
        for y in (0..small.height()).step_by(2) {
            for x in 0..small.width() {
                let t = small.get_pixel(x, y);
                let b = if y + 1 < small.height() { *small.get_pixel(x, y + 1) } else { *t };
                let _ = write!(out, "\x1b[38;2;{};{};{}m\x1b[48;2;{};{};{}m▀", t[0], t[1], t[2], b[0], b[1], b[2]);
            }
            let _ = writeln!(out, "\x1b[0m");
        }
    }
    let _ = out.flush();
}

/// A row of colour swatches, two cells per colour; fully transparent entries show as `··`.
pub fn swatches(colors: &[[u8; 4]]) -> String {
    let mut s = String::new();
    for c in colors {
        if c[3] == 0 {
            s.push_str("··");
        } else {
            s.push_str(&format!("\x1b[48;2;{};{};{}m  \x1b[0m", c[0], c[1], c[2]));
        }
        s.push(' ');
    }
    s
}

pub fn heading(title: &str) {
    println!("\n{}", title.bold().underlined());
}

pub fn info(text: &str) {
    println!("{}", text.dim());
}

/// Read one line; end of input counts as quitting.
fn read_line(prompt: &str) -> Result<String, Nav> {
    print!("{prompt}");
    let _ = io::stdout().flush();
    let mut line = String::new();
    match io::stdin().lock().read_line(&mut line) {
        Ok(0) | Err(_) => {
            println!();
            Err(Nav::Quit)
        }
        Ok(_) => Ok(line.trim().to_string()),
    }
}

fn nav_hint(can_back: bool) -> &'static str {
    if can_back { "b back, q quit" } else { "q quit" }
}

/// Numbered choice; Enter takes `default`.
pub fn choose(title: &str, options: &[String], default: usize, can_back: bool) -> Result<usize, Nav> {
    heading(title);
    for (i, o) in options.iter().enumerate() {
        let n = format!("{:>3})", i + 1);
        if i == default {
            println!("{} {}", n.bold(), o.as_str().bold());
        } else {
            println!("{n} {o}");
        }
    }
    loop {
        let line = read_line(&format!("Choice [{}] ({}): ", default + 1, nav_hint(can_back)))?;
        match line.as_str() {
            "" => return Ok(default),
            "q" => return Err(Nav::Quit),
            "b" if can_back => return Err(Nav::Back),
            _ => match line.parse::<usize>() {
                Ok(n) if (1..=options.len()).contains(&n) => return Ok(n - 1),
                _ => println!("Enter a number from 1 to {}.", options.len()),
            },
        }
    }
}

/// Free-form answer checked by `parse`; Enter takes `default`.
pub fn ask<T>(prompt: &str, default: &str, parse: impl Fn(&str) -> Result<T, String>) -> Result<T, Nav> {
    loop {
        let line = read_line(&format!("{prompt} [{default}] (b back, q quit): "))?;
        let line = match line.as_str() {
            "" => default,
            "q" => return Err(Nav::Quit),
            "b" => return Err(Nav::Back),
            s => s,
        };
        match parse(line) {
            Ok(v) => return Ok(v),
            Err(e) => println!("{e}"),
        }
    }
}

/// Integer in `lo..=hi`.
pub fn ask_int(prompt: &str, lo: i32, hi: i32, default: i32) -> Result<i32, Nav> {
    ask(&format!("{prompt} ({lo}–{hi})"), &default.to_string(), |s| match s.parse::<i32>() {
        Ok(v) if (lo..=hi).contains(&v) => Ok(v),
        _ => Err(format!("Enter a whole number from {lo} to {hi}.")),
    })
}

pub fn ask_bool(prompt: &str, default: bool) -> Result<bool, Nav> {
    ask(prompt, if default { "y" } else { "n" }, |s| match s.to_ascii_lowercase().as_str() {
        "y" | "yes" => Ok(true),
        "n" | "no" => Ok(false),
        _ => Err("Answer y or n.".into()),
    })
}
