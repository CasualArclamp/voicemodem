//! The window: lines, the transmitter's controls, the push-to-talk button,
//! and what the receiver hears.

use std::sync::mpsc::Sender;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use eframe::egui::{
    self, Align2, Color32, FontId, Painter, Pos2, Rect, RichText, Sense, Stroke, Ui, pos2, vec2,
};
use modem::{Profile, VoiceMode};

use crate::engine::{Command, Status};
use crate::settings::Settings;

const BACKDROP: Color32 = Color32::from_rgb(12, 14, 18);
const GRID: Color32 = Color32::from_rgb(40, 46, 56);
const TRACE: Color32 = Color32::from_rgb(120, 220, 160);
const LABEL: Color32 = Color32::from_rgb(150, 160, 175);
const ON_AIR: Color32 = Color32::from_rgb(190, 40, 40);

/// Width of the station panel on the left.
const PANEL_W: f32 = 330.0;

/// Highest frequency the spectrum shows.
const DISPLAY_HZ: f64 = 4000.0;

pub struct VoiceApp {
    commands: Sender<Command>,
    status: Arc<Mutex<Status>>,
    view: Status,
    settings: Settings,
    inputs: Vec<String>,
    outputs: Vec<String>,
    /// What the engine was last told about the transmitter's key, and the
    /// latch's state when latching.
    keyed: bool,
    latched_on: bool,
    /// Whether the constellation is also drawn large, in a window of its own.
    constellation_open: bool,
    /// A demonstration: when it started, and whether it is talking now.
    demo: Option<Instant>,
    demo_talking: bool,
}

/// A demonstration talks for this long in every cycle of this long.
const DEMO_TALK: f64 = 10.0;
const DEMO_CYCLE: f64 = 15.0;

impl std::fmt::Debug for VoiceApp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VoiceApp").field("settings", &self.settings).finish_non_exhaustive()
    }
}

impl VoiceApp {
    /// The window; with `demo`, on the loopback, talking to itself.
    pub fn new(commands: Sender<Command>, status: Arc<Mutex<Status>>, settings: Settings, demo: bool) -> Self {
        let app = Self {
            commands,
            status,
            view: Status::default(),
            settings,
            inputs: line::input_devices(),
            outputs: line::output_devices(),
            keyed: false,
            latched_on: false,
            constellation_open: false,
            demo: demo.then(Instant::now),
            demo_talking: false,
        };
        app.send(Command::Mode(app.mode()));
        app.send(Command::Text(app.settings.text.clone()));
        app.send(Command::Level(app.settings.level));
        app.send(Command::FullDuplex(app.settings.full_duplex));
        if demo {
            // The loopback, and the mic and speaker if they have been chosen:
            // with no speaker it is still something to look at.
            let operator = (!app.settings.mic.is_empty() && !app.settings.speaker.is_empty())
                .then(|| (app.settings.mic.clone(), app.settings.speaker.clone()));
            app.send(Command::Open { radio: None, loopback_snr: app.settings.loopback_snr, operator });
        }
        app
    }

    /// A demonstration's next move: synthetic speech for [`DEMO_TALK`]
    /// seconds of every [`DEMO_CYCLE`].
    fn demonstrate(&mut self) {
        let Some(started) = self.demo else { return };
        let into = started.elapsed().as_secs_f64() % DEMO_CYCLE;
        let talk = (0.5..0.5 + DEMO_TALK).contains(&into);
        if talk && !self.demo_talking {
            self.send(Command::Speak(voice::synthetic_speech(DEMO_TALK, modem::profile::FS)));
            self.send(Command::Ptt(true));
        } else if !talk && self.demo_talking {
            self.send(Command::Ptt(false));
        }
        self.demo_talking = talk;
    }

    fn send(&self, command: Command) {
        let _ = self.commands.send(command);
    }

    fn mode(&self) -> &'static VoiceMode {
        VoiceMode::by_name(&self.settings.mode).unwrap_or(VoiceMode::robust(Profile::Narrow))
    }

    fn changed(&self) {
        self.settings.save();
    }

    /// The row of lines: the radio's devices or the loopback, and the
    /// operator's.
    fn lines(&mut self, ui: &mut Ui) {
        ui.horizontal_wrapped(|ui| {
            ui.label(RichText::new("radio").strong());
            if ui.checkbox(&mut self.settings.loopback, "loopback, no radio").changed() {
                self.changed();
            }
            if self.settings.loopback {
                ui.label("SNR");
                if ui
                    .add(egui::Slider::new(&mut self.settings.loopback_snr, -2.0..=40.0).suffix(" dB"))
                    .on_hover_text("Es/N0 of the noise added between the transmitter and the receiver")
                    .changed()
                {
                    self.changed();
                }
            } else {
                device(ui, "in", &mut self.settings.radio_in, &self.inputs, "radio in");
                device(ui, "out", &mut self.settings.radio_out, &self.outputs, "radio out");
            }
            ui.separator();
            ui.label(RichText::new("operator").strong());
            device(ui, "mic", &mut self.settings.mic, &self.inputs, "mic");
            device(ui, "speaker", &mut self.settings.speaker, &self.outputs, "speaker");
            ui.separator();
            if ui.button("Open").on_hover_text("Open the lines chosen, closing any that are open").clicked() {
                let radio = (!self.settings.loopback)
                    .then(|| (self.settings.radio_in.clone(), self.settings.radio_out.clone()));
                let operator = (!self.settings.mic.is_empty() && !self.settings.speaker.is_empty())
                    .then(|| (self.settings.mic.clone(), self.settings.speaker.clone()));
                self.send(Command::Open { radio, loopback_snr: self.settings.loopback_snr, operator });
                self.changed();
            }
            if ui.button("Close").clicked() {
                self.send(Command::Close);
            }
            if ui.button("⟳").on_hover_text("Look for audio devices again").clicked() {
                self.inputs = line::input_devices();
                self.outputs = line::output_devices();
            }
        });
        ui.label(RichText::new(&self.view.lines).color(LABEL).small());
    }

    fn transmitter(&mut self, ui: &mut Ui) {
        ui.label(RichText::new("transmit").strong());
        let mode = self.mode();
        let mut profile = mode.profile;
        ui.horizontal(|ui| {
            for p in Profile::ALL {
                ui.radio_value(&mut profile, p, p.label());
            }
        });
        let mut chosen = mode;
        if profile != mode.profile {
            chosen = VoiceMode::robust(profile);
        }
        egui::ComboBox::from_id_salt("mode")
            .width(PANEL_W - 30.0)
            .selected_text(chosen.name)
            .show_ui(ui, |ui| {
                for m in VoiceMode::of_profile(profile) {
                    // Asked each time the list opens, so that weights put in
                    // place while the program runs are seen.
                    let missing = voice::available(m.codec).err();
                    let label = format!(
                        "{}  {} {}, {}{}",
                        m.name,
                        m.modulation.label(),
                        m.rate.label(),
                        m.codec.label(),
                        if missing.is_some() { " (needs weights)" } else { "" }
                    );
                    let item = ui.selectable_value(&mut chosen, m, label);
                    if let Some(why) = missing {
                        item.on_hover_text(why);
                    }
                }
            });
        if chosen != mode {
            self.settings.mode = chosen.name.to_string();
            self.send(Command::Mode(chosen));
            self.changed();
        }
        let (low, high) = chosen.profile.band();
        ui.label(
            RichText::new(format!(
                "{} {} at {:.0} baud, {:.0}-{:.0} Hz\n{} ({:.0} bit/s), {:.2} s a codeword",
                chosen.modulation.label(),
                chosen.rate.label(),
                chosen.profile.baud(),
                low,
                high,
                chosen.codec.label(),
                chosen.codec.bit_rate(),
                chosen.codeword_seconds()
            ))
            .color(LABEL)
            .small(),
        );
        ui.horizontal(|ui| {
            ui.label("text");
            let edit = ui
                .add(egui::TextEdit::singleline(&mut self.settings.text).desired_width(PANEL_W - 80.0).char_limit(64))
                .on_hover_text("Sent a character a codeword, over and over: your callsign");
            if edit.changed() {
                self.send(Command::Text(self.settings.text.clone()));
                self.changed();
            }
        });
        ui.horizontal(|ui| {
            ui.label("level");
            if ui
                .add(egui::Slider::new(&mut self.settings.level, -36.0..=-3.0).suffix(" dBFS"))
                .on_hover_text("Transmit level, rms. Keep the rig's ALC quiet: PSK wants a linear chain.")
                .changed()
            {
                self.send(Command::Level(self.settings.level));
                self.changed();
            }
        });
        if ui
            .checkbox(&mut self.settings.full_duplex, "decode while transmitting")
            .on_hover_text("For a satellite, whose downlink carries your own signal back")
            .changed()
        {
            self.send(Command::FullDuplex(self.settings.full_duplex));
            self.changed();
        }
    }

    fn push_to_talk(&mut self, ui: &mut Ui) {
        ui.add_space(8.0);
        let text = if self.view.transmitting { "ON AIR" } else { "PUSH TO TALK" };
        let fill = if self.keyed || self.view.transmitting { ON_AIR } else { Color32::from_rgb(45, 52, 64) };
        let button = ui.add_sized(
            [PANEL_W - 20.0, 72.0],
            egui::Button::new(RichText::new(text).size(22.0).strong()).fill(fill),
        );
        let space = !ui.ctx().egui_wants_keyboard_input() && ui.input(|i| i.key_down(egui::Key::Space));
        if self.settings.latch && (button.clicked() || ui.input(|i| i.key_pressed(egui::Key::Space)) && !ui.ctx().egui_wants_keyboard_input()) {
            self.latched_on = !self.latched_on;
        }
        let want = if self.settings.latch { self.latched_on } else { button.is_pointer_button_down_on() || space };
        let want = want && self.view.open;
        if want != self.keyed {
            self.keyed = want;
            self.send(Command::Ptt(want));
        }
        ui.horizontal(|ui| {
            if ui.checkbox(&mut self.settings.latch, "latch").on_hover_text("A click keys, the next unkeys").changed() {
                self.latched_on = false;
                self.changed();
            }
            ui.label(RichText::new("or hold Space").color(LABEL).small());
        });
        meter(ui, "mic", self.view.mic_level_dbfs);
        if self.view.transmitting {
            ui.label(format!("{:.1} s still to go out", self.view.behind));
        }
    }

    fn receiver(&self, ui: &mut Ui) {
        ui.add_space(8.0);
        ui.label(RichText::new("receive").strong());
        ui.label(&self.view.receiving);
        let v = &self.view;
        let figure = |x: Option<f64>, unit: &str, places: usize| x.map_or("-".to_string(), |x| format!("{x:+.places$} {unit}"));
        egui::Grid::new("rx").num_columns(2).show(ui, |ui| {
            ui.label("mode");
            ui.label(v.rx_mode.map_or("-", |m| m.name));
            ui.end_row();
            ui.label("Es/N0");
            ui.label(v.snr_db.map_or("-".to_string(), |s| format!("{s:.1} dB")));
            ui.end_row();
            ui.label("carrier");
            ui.label(figure(v.offset_hz, "Hz", 1));
            ui.end_row();
            ui.label("clock");
            ui.label(figure(v.drift_ppm, "ppm", 0));
            ui.end_row();
            ui.label("codewords");
            ui.label(format!("{} heard, {} lost", v.heard, v.lost));
            ui.end_row();
            ui.label("text");
            ui.label(RichText::new(&v.text).monospace());
            ui.end_row();
        });
        meter(ui, "radio", v.rx_level_dbfs);
    }

    /// What the scope says along its bottom, and how good the signal is on a
    /// scale of nought to one, if it has been measured.
    fn scope_label(&self) -> (String, Option<f32>) {
        let v = &self.view;
        let what = v.modulation.map_or("", |m| m.label());
        match (v.modulation, v.scope_snr_db) {
            (Some(m), Some(snr)) => {
                // Against what the modulation needs: red at its threshold,
                // green ten decibels above it.
                let need = match m.bits() {
                    1 => 4.0,
                    2 => 7.0,
                    _ => 12.0,
                };
                (format!("{what}  Es/N0 {snr:.1} dB"), Some(((snr - need) / 10.0).clamp(0.0, 1.0) as f32))
            }
            _ => (what.to_string(), None),
        }
    }

    fn scopes(&mut self, ui: &mut Ui) {
        let height = (ui.available_height() * 0.5).clamp(180.0, 420.0);
        let (label, quality) = self.scope_label();
        ui.horizontal(|ui| {
            let scope = symbol_scope(ui, &self.view.points, &label, quality, height);
            if scope.on_hover_text("Click to draw it large").clicked() {
                self.constellation_open = !self.constellation_open;
            }
            ui.vertical(|ui| {
                spectrum(ui, &self.view.spectrum, self.view.hz_per_bin, height, self.mode().profile);
            });
        });
        let mut open = self.constellation_open;
        egui::Window::new("constellation").open(&mut open).resizable(true).default_size([620.0, 640.0]).show(
            ui.ctx(),
            |ui| {
                let side = ui.available_width().min(ui.available_height()).max(240.0);
                symbol_scope(ui, &self.view.points, &label, quality, side);
            },
        );
        self.constellation_open = open;
        ui.add_space(6.0);
        ui.label(RichText::new("log").strong());
        egui::ScrollArea::vertical().stick_to_bottom(true).auto_shrink([false, false]).show(ui, |ui| {
            for line in &self.view.log {
                ui.label(RichText::new(line).monospace().size(11.0));
            }
        });
    }
}

impl eframe::App for VoiceApp {
    fn ui(&mut self, ui: &mut Ui, _frame: &mut eframe::Frame) {
        if let Ok(status) = self.status.lock() {
            self.view = status.clone();
        }
        ui.ctx().request_repaint_after(Duration::from_millis(50));
        self.demonstrate();

        egui::Panel::top("lines").show(ui, |ui| {
            ui.add_space(4.0);
            self.lines(ui);
            ui.add_space(4.0);
        });
        egui::Panel::left("station").resizable(false).exact_size(PANEL_W).show(ui, |ui| {
            ui.add_space(6.0);
            self.transmitter(ui);
            self.push_to_talk(ui);
            ui.separator();
            self.receiver(ui);
        });
        egui::CentralPanel::default().show(ui, |ui| self.scopes(ui));
    }
}

/// A combo box of device names, with nothing chosen shown as such.
fn device(ui: &mut Ui, label: &str, chosen: &mut String, names: &[String], id: &str) {
    ui.label(label);
    let shown = if chosen.is_empty() { "(choose)".to_string() } else { chosen.clone() };
    egui::ComboBox::from_id_salt(id).width(170.0).selected_text(shown).show_ui(ui, |ui| {
        for name in names {
            ui.selectable_value(chosen, name.clone(), name);
        }
    });
}

/// A level bar from -60 to 0 dBFS.
fn meter(ui: &mut Ui, label: &str, dbfs: f64) {
    ui.horizontal(|ui| {
        ui.label(RichText::new(format!("{label:<6}")).monospace());
        let (rect, painter) = allocate(ui, vec2(PANEL_W - 110.0, 12.0));
        painter.rect_filled(rect, 2.0, BACKDROP);
        let t = ((dbfs + 60.0) / 60.0).clamp(0.0, 1.0) as f32;
        let colour = if dbfs > -3.0 { ON_AIR } else { TRACE };
        painter.rect_filled(Rect::from_min_size(rect.min, vec2(rect.width() * t, rect.height())), 2.0, colour);
        ui.label(RichText::new(format!("{dbfs:>5.0} dB")).monospace().small());
    });
}

fn allocate(ui: &mut Ui, size: egui::Vec2) -> (Rect, Painter) {
    let (response, painter) = ui.allocate_painter(size, Sense::hover());
    (response.rect, painter)
}

fn border(painter: &Painter, rect: Rect) {
    painter.rect_stroke(rect, 0.0, Stroke::new(1.0, GRID), egui::StrokeKind::Inside);
}

/// The receiver's points, drawn as BinModem draws a constellation
/// (`crates/gui/src/scopes.rs`, `symbol_scope`): blue axes, a tick at each
/// arm tip where an ideal symbol lands, and every symbol a small faint
/// square, one colour, so that the symbols landing on a point build up into
/// it as one mesh rather than thousands of shapes. What a reader wants from
/// it is the shape of the clusters, and that is what building up shows.
fn symbol_scope(ui: &mut Ui, points: &[[f32; 2]], label: &str, quality: Option<f32>, size: f32) -> egui::Response {
    let (response, painter) = ui.allocate_painter(vec2(size, size), Sense::click());
    let rect = response.rect;
    painter.rect_filled(rect, 0.0, Color32::BLACK);

    let centre = rect.center();
    // Square, so the two axes share a scale.
    let radius = (rect.width().min(rect.height()) * 0.5) - 12.0;
    let axis = Color32::from_rgb(70, 130, 200);
    painter.line_segment([pos2(centre.x - radius, centre.y), pos2(centre.x + radius, centre.y)], Stroke::new(1.5, axis));
    painter.line_segment([pos2(centre.x, centre.y - radius), pos2(centre.x, centre.y + radius)], Stroke::new(1.5, axis));
    for dx in [-1.0f32, 1.0] {
        let x = centre.x + dx * radius;
        painter.line_segment([pos2(x, centre.y - 5.0), pos2(x, centre.y + 5.0)], Stroke::new(1.0, axis.gamma_multiply(0.8)));
    }

    // A unit-magnitude symbol sits at the arm tip; PSK's points all do.
    let at = |re: f32, im: f32| pos2(centre.x + re.clamp(-1.4, 1.4) * radius, centre.y - im.clamp(-1.4, 1.4) * radius);
    let mut mesh = egui::Mesh::default();
    let crowded = points.len() > 200;
    let side = (radius / 180.0).clamp(1.0, 2.5) * if crowded { 1.0 } else { 1.7 };
    let colour = Color32::from_rgba_unmultiplied(120, 220, 160, if crowded { 110 } else { 150 });
    for p in points {
        mesh.add_colored_rect(Rect::from_center_size(at(p[0], p[1]), vec2(side, side)), colour);
    }
    painter.add(egui::Shape::mesh(mesh));

    // Always say something: a silent, empty scope gives no way to tell a
    // receiver that is not decoding from a display that is not being fed.
    let grey = Color32::from_rgb(150, 160, 175);
    let (text, colour) = match quality {
        Some(q) => (label.to_string(), margin_colour(q)),
        None if !points.is_empty() => (format!("{label}  {} points", points.len()), grey),
        None => (format!("{label}  no symbols").trim_start().to_string(), Color32::from_rgb(120, 100, 100)),
    };
    painter.text(pos2(rect.left() + 6.0, rect.bottom() - 4.0), Align2::LEFT_BOTTOM, text, FontId::monospace(11.0), colour);
    border(&painter, rect);
    response
}

/// Green at a full margin, through yellow, to red at the threshold: BinModem's
/// colouring of its quality figure.
fn margin_colour(margin: f32) -> Color32 {
    let m = margin.clamp(0.0, 1.0);
    if m > 0.5 {
        let t = (m - 0.5) / 0.5;
        Color32::from_rgb(
            (255.0 * (1.0 - t) + 60.0 * t) as u8,
            (215.0 * (1.0 - t) + 230.0 * t) as u8,
            (60.0 * (1.0 - t) + 90.0 * t) as u8,
        )
    } else {
        let t = m / 0.5;
        Color32::from_rgb(
            (235.0 * (1.0 - t) + 255.0 * t) as u8,
            (60.0 * (1.0 - t) + 215.0 * t) as u8,
            (55.0 * (1.0 - t) + 60.0 * t) as u8,
        )
    }
}

/// The radio's receive audio, 0 to 4 kHz, with the profile's band marked.
fn spectrum(ui: &mut Ui, bins: &[f32], hz_per_bin: f64, height: f32, profile: Profile) {
    let (rect, painter) = allocate(ui, vec2(ui.available_width(), height));
    painter.rect_filled(rect, 0.0, BACKDROP);
    let (floor, ceiling) = (-110.0f32, -10.0f32);
    let x_of = |hz: f64| rect.left() + rect.width() * (hz / DISPLAY_HZ) as f32;
    let (low, high) = profile.band();
    painter.rect_filled(
        Rect::from_min_max(pos2(x_of(low), rect.top()), pos2(x_of(high), rect.bottom())),
        0.0,
        Color32::from_rgba_premultiplied(40, 60, 90, 60),
    );
    let mut db = ceiling;
    while db >= floor {
        let y = rect.top() + rect.height() * (ceiling - db) / (ceiling - floor);
        painter.line_segment([pos2(rect.left(), y), pos2(rect.right(), y)], Stroke::new(1.0, GRID));
        painter.text(pos2(rect.left() + 3.0, y), Align2::LEFT_CENTER, format!("{db:.0}"), FontId::monospace(9.0), LABEL);
        db -= 20.0;
    }
    if hz_per_bin > 0.0 && !bins.is_empty() {
        let w = rect.width().max(1.0) as usize;
        let points: Vec<Pos2> = (0..w)
            .map(|x| {
                let hz = DISPLAY_HZ * x as f64 / w as f64;
                let v = bins.get((hz / hz_per_bin).round() as usize).copied().unwrap_or(floor);
                let t = ((v - floor) / (ceiling - floor)).clamp(0.0, 1.0);
                pos2(rect.left() + x as f32, rect.bottom() - rect.height() * t)
            })
            .collect();
        for pair in points.windows(2) {
            painter.line_segment([pair[0], pair[1]], Stroke::new(1.2, TRACE));
        }
    }
    let mut hz = 0.0;
    while hz <= DISPLAY_HZ {
        painter.text(pos2(x_of(hz), rect.bottom() - 2.0), Align2::CENTER_BOTTOM, format!("{hz:.0}"), FontId::monospace(9.0), LABEL);
        hz += 500.0;
    }
    painter.text(rect.right_top() + vec2(-4.0, 4.0), Align2::RIGHT_TOP, "radio audio", FontId::monospace(10.0), LABEL);
    border(&painter, rect);
}
