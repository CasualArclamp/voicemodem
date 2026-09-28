//! voicemodem: digital voice through a radio's voice channel.
//!
//! ```text
//!   voicemodem                    the window: talk and listen through a radio
//!   voicemodem tx|rx|selftest     headless; `voicemodem help` for the rest
//! ```

mod cli;
mod compare;
mod engine;
mod gui;
mod settings;
mod speech;

use modem::{Profile, VoiceMode};

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let result = match args.first().map(String::as_str) {
        Some("demo") => window(true),
        _ => match cli::run(&args) {
            Some(result) => result,
            None => window(false),
        },
    };
    if let Err(e) = result {
        eprintln!("voicemodem: {e}");
        std::process::exit(1);
    }
}

/// The window; in a demonstration, on the loopback and talking to itself.
fn window(demo: bool) -> Result<(), String> {
    let settings = settings::Settings::load();
    let mode = VoiceMode::by_name(&settings.mode).unwrap_or(VoiceMode::robust(Profile::Narrow));
    let (commands, status) = engine::spawn(mode);
    let title = format!("voicemodem {}", env!("CARGO_PKG_VERSION"));
    let options = eframe::NativeOptions {
        viewport: eframe::egui::ViewportBuilder::default()
            .with_inner_size([1180.0, 820.0])
            .with_min_inner_size([900.0, 620.0])
            .with_title(&title),
        ..Default::default()
    };
    let quit = commands.clone();
    let result = eframe::run_native(
        &title,
        options,
        Box::new(move |cc| {
            cc.egui_ctx.set_visuals(eframe::egui::Visuals::dark());
            Ok(Box::new(gui::VoiceApp::new(commands, status, settings, demo)))
        }),
    );
    let _ = quit.send(engine::Command::Quit);
    result.map_err(|e| e.to_string())
}
