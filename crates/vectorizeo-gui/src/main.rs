//! `vectorizeo-gui`: desktop and web front-end for the vectorizeo tracer — the whole workflow
//! (classify, palette, segmentation, smoothing, export) in one window with a canvas and a settings
//! panel. The web build is made with trunk (`index.html` next to this crate's `Cargo.toml`).

mod app;
mod canvas;
mod edit;
mod help;
mod inspector;
mod theme;
mod worker;

#[cfg(not(target_arch = "wasm32"))]
fn main() -> eframe::Result {
    use eframe::egui;

    let input = std::env::args_os().nth(1).map(std::path::PathBuf::from);
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_title("vectorizeo")
            .with_icon(window_icon())
            .with_app_id("vectorizeo-gui")
            .with_inner_size([1280.0, 820.0])
            .with_min_inner_size([760.0, 480.0])
            .with_drag_and_drop(true),
        #[cfg(feature = "wgpu")]
        renderer: eframe::Renderer::Wgpu,
        #[cfg(not(feature = "wgpu"))]
        renderer: eframe::Renderer::Glow,
        ..Default::default()
    };
    eframe::run_native("vectorizeo", options, Box::new(|cc| Ok(Box::new(app::App::new(cc, input)))))
}

/// The window icon: `assets/icon.png` (the small-size logo on an accent tile).
#[cfg(not(target_arch = "wasm32"))]
fn window_icon() -> eframe::egui::IconData {
    let img = image::load_from_memory(include_bytes!("../assets/icon.png"))
        .expect("assets/icon.png is a valid PNG")
        .into_rgba8();
    let (width, height) = img.dimensions();
    eframe::egui::IconData { rgba: img.into_raw(), width, height }
}

/// Web build (`trunk serve`): run in the `<canvas id="vectorizeo">` of `index.html`.
#[cfg(target_arch = "wasm32")]
fn main() {
    use eframe::wasm_bindgen::JsCast;
    use eframe::web_sys;

    let document = web_sys::window().and_then(|w| w.document()).expect("a document");
    let canvas = document
        .get_element_by_id("vectorizeo")
        .and_then(|e| e.dyn_into::<web_sys::HtmlCanvasElement>().ok())
        .expect("<canvas id=\"vectorizeo\"> in index.html");
    wasm_bindgen_futures::spawn_local(async move {
        let started = eframe::WebRunner::new()
            .start(canvas, eframe::WebOptions::default(), Box::new(|cc| Ok(Box::new(app::App::new(cc, None)))))
            .await;
        if let Some(loading) = document.get_element_by_id("loading") {
            match started {
                Ok(()) => loading.remove(),
                Err(e) => loading.set_text_content(Some(&format!("vectorizeo could not start: {e:?}"))),
            }
        }
    });
}
