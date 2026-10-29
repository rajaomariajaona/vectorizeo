<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/logo-white.svg">
    <img src="assets/logo.svg" alt="Vectorizeo logo" width="120">
  </picture>
</p>

# Vectorizeo

Vectorizeo turns bitmap images (PNG, JPEG, BMP, GIF, TIFF) into clean vector graphics. It is written in Rust and comes as a command-line tool, a terminal wizard and a desktop/web GUI.

## Features

- Full tracing pipeline: classification, palette search, pixel segmentation (with anti-aliasing aware sub-pixel segmentation), boundary tracing, contour smoothing and Bézier fitting.
- SVG output, plus EPS.
- Presets for photos, anti-aliased artwork and artwork without anti-aliasing, at three detail levels; every setting can also be tuned by hand.
- Fixed or automatically proposed palettes, transparency handling, shape modes (stacked, cut-outs, grouped).
- GUI with live stage-by-stage results, manual segmentation editing and background removal. It also runs in the browser (WebAssembly).

A web version is available at [vectorizeo.pages.dev](https://vectorizeo.pages.dev). It runs in WebAssembly inside the browser, so it will be slow on anything other than a PC; the desktop GUI is faster.

## Layout

| crate | purpose |
|---|---|
| `vectorizeo-core` | the engine: stages, configuration, SVG and EPS writers |
| `vectorizeo-pipeline` | pipeline steps and session state shared by the front-ends |
| `vectorizeo-cli` | the `vectorizeo` command (`trace`, `segment`, `palette`, `interactive`) |
| `vectorizeo-gui` | desktop and web GUI built with egui |

## Install

**Linux** (x86_64):

```sh
curl -fsSL https://raw.githubusercontent.com/rajaomariajaona/vectorizeo/main/install.sh | bash
```

This installs `vectorizeo` and `vectorizeo-gui` into `~/.local/bin`, adds a desktop entry and shell completions. Set `VECTORIZEO_VERSION=v0.1.0` to pin a release, or run it with `bash -s -- --uninstall` to remove everything.

**Portable packages**: `vectorizeo-x86_64-linux.tar.gz` and `vectorizeo-x86_64-windows.zip` on the [releases page](https://github.com/rajaomariajaona/vectorizeo/releases). Unpack anywhere and run; nothing else is needed.

**Releasing**: bump the version in the crates' `Cargo.toml`, then `git tag v0.1.0 && git push --tags`. The release workflow builds both packages and publishes them with a `SHA256SUMS` file.

## Usage

```sh
# Trace an image
cargo run --release -p vectorizeo -- trace input.png output.svg

# Choose the image type, detail level and palette
cargo run --release -p vectorizeo -- trace logo.png logo.svg -t blended -q high -c auto

# Step through the settings in the terminal
cargo run --release -p vectorizeo -- interactive input.png

# Desktop GUI
cargo run --release -p vectorizeo-gui -- input.png
```

`just --list` shows the build, test, install and web-build recipes. The web GUI needs [trunk](https://trunkrs.dev) and the `wasm32-unknown-unknown` target (`just web-setup`).

## Development

```sh
cargo test --workspace
cargo fmt --all
```

The workspace uses Rust edition 2024.

## License

Vectorizeo is free software, released under the [GNU General Public License v3.0 or later](LICENSE).
