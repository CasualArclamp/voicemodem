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
        Some("demo") => cli::demo(&args[1..]).and_then(|demo| window(Some(demo))),
        _ => match cli::run(&args) {
            Some(result) => result,
            None => window(None),
        },
    };
    if let Err(e) = result {
        eprintln!("voicemodem: {e}");
        std::process::exit(1);
    }
}

/// The window; in a demonstration, on the loopback and talking to itself.
fn window(demo: Option<cli::Demo>) -> Result<(), String> {
    let path = demo.as_ref().and_then(|d| d.picture.clone());
    // A picture is for showing to others, so it starts from the defaults:
    // none of this machine's devices or its operator's callsign are in it.
    let mut settings = if path.is_some() { settings::Settings::default() } else { settings::Settings::load() };
    if let Some(d) = &demo {
        if let Some(mode) = d.mode {
            settings.mode = mode.name.to_string();
        }
        if let Some(snr) = d.snr {
            settings.loopback_snr = snr;
        }
        if let Some(text) = &d.text {
            settings.text.clone_from(text);
        }
    }
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
    let picture = path.clone().map(gui::Picture::new);
    let taken = picture.as_ref().map(gui::Picture::outcome);
    let is_demo = demo.is_some();
    let quit = commands.clone();
    let result = eframe::run_native(
        &title,
        options,
        Box::new(move |cc| {
            cc.egui_ctx.set_visuals(eframe::egui::Visuals::dark());
            Ok(Box::new(gui::VoiceApp::new(commands, status, settings, is_demo, picture)))
        }),
    );
    let _ = quit.send(engine::Command::Quit);
    result.map_err(|e| e.to_string())?;
    let (Some(taken), Some(path)) = (taken, path) else { return Ok(()) };
    let outcome = taken.lock().ok().and_then(|mut outcome| outcome.take());
    outcome.unwrap_or_else(|| Err("the window closed before its picture was taken".into()))?;
    println!("wrote {}", path.display());
    Ok(())
}
