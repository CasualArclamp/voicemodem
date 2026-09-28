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
    /// Whether the engine was last told to open the lines rather than close
    /// them; while it was, a device chosen takes effect at once. Not the
    /// view's `open`, which is false while no line would open: just when
    /// choosing another device ought to try it.
    lines_open: bool,
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
            lines_open: demo,
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
            let operator = chosen_operator(&app.settings);
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
        // As the settings were before this frame's widgets, to tell once they
        // are drawn which line, if either, a click has changed.
        let before = self.settings.clone();
        ui.horizontal_wrapped(|ui| {
            ui.label(RichText::new("radio").strong());
            ui.checkbox(&mut self.settings.loopback, "loopback, no radio");
            if self.settings.loopback {
                ui.label("SNR");
                if ui
                    .add(egui::Slider::new(&mut self.settings.loopback_snr, -2.0..=40.0).suffix(" dB"))
                    .on_hover_text("Es/N0 of the noise added between the transmitter and the receiver")
                    .changed()
                {
                    // Straight into the running loopback: before, it was read
                    // only when the lines were opened.
                    self.send(Command::LoopbackSnr(self.settings.loopback_snr));
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
                let radio = chosen_radio(&self.settings);
                let operator = chosen_operator(&self.settings);
                self.send(Command::Open { radio, loopback_snr: self.settings.loopback_snr, operator });
                self.lines_open = true;
                self.changed();
            }
            if ui.button("Close").clicked() {
                self.send(Command::Close);
                self.lines_open = false;
            }
            if ui.button("⟳").on_hover_text("Look for audio devices again").clicked() {
                self.inputs = line::input_devices();
                self.outputs = line::output_devices();
            }
        });
        if self.settings != before {
            self.changed();
            if self.lines_open {
                for command in switched(&before, &self.settings) {
                    self.send(command);
                }
            }
        }
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
            // The listener keeps 64 characters, more than the panel has room
            // for, so the row shows the last 16, which move along like a
            // ticker as each codeword brings another. What has scrolled off
            // is still there to read, by hovering.
            let shown = last_chars(&v.text, 16);
            let row = ui.label(RichText::new(shown).monospace());
            if shown.len() < v.text.len() {
                row.on_hover_text(RichText::new(&v.text).monospace());
            }
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

/// The radio's line as `settings` choose it: its two devices, or None for
/// the loopback. A device not chosen yet goes as an empty name, which the
/// engine waits on rather than guessing at.
fn chosen_radio(settings: &Settings) -> Option<(String, String)> {
    (!settings.loopback).then(|| (settings.radio_in.clone(), settings.radio_out.clone()))
}

/// The operator's line as `settings` choose it: the mic and the speaker, or
/// None until both are chosen.
fn chosen_operator(settings: &Settings) -> Option<(String, String)> {
    (!settings.mic.is_empty() && !settings.speaker.is_empty())
        .then(|| (settings.mic.clone(), settings.speaker.clone()))
}

/// What open lines are to be told when the settings go from `before` to
/// `now`: the line whose devices changed, or the radio's when the loopback
/// is ticked or unticked, reopened on its own, so that the other and
/// whatever is being sent or heard carry on.
fn switched(before: &Settings, now: &Settings) -> Vec<Command> {
    let mut commands = Vec::new();
    let radio = chosen_radio(now);
    if radio != chosen_radio(before) {
        commands.push(Command::Radio(radio));
    }
    let operator = chosen_operator(now);
    if operator != chosen_operator(before) {
        commands.push(Command::Operator(operator));
    }
    commands
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

/// The last `n` characters of `text`, or all of it when it is shorter. It
/// counts characters rather than bytes: the transmitter sends only printable
/// ASCII, but the listener makes a character of each byte it hears, and one
/// from 0x80 up is two bytes in the string; slicing between those would panic.
fn last_chars(text: &str, n: usize) -> &str {
    let start = text.char_indices().rev().take(n).last().map_or(text.len(), |(i, _)| i);
    &text[start..]
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
    // Square, so the two axes share a scale. A unit-magnitude symbol -- every
    // PSK point -- sits at the arm tips, which are drawn well inside the
    // edge: BinModem puts them 12 pixels from it, which suits V.34's grid,
    // and here left noisy PSK symbols falling off the scope. With the tips
    // at 1/REACH of the way out, a symbol blown out to REACH still lands
    // inside, and one blown further is pinned to the edge rather than lost.
    const REACH: f32 = 1.6;
    let half = rect.width().min(rect.height()) * 0.5 - 4.0;
    let radius = half / REACH;
    let axis = Color32::from_rgb(70, 130, 200);
    painter.line_segment([pos2(centre.x - half, centre.y), pos2(centre.x + half, centre.y)], Stroke::new(1.5, axis));
    painter.line_segment([pos2(centre.x, centre.y - half), pos2(centre.x, centre.y + half)], Stroke::new(1.5, axis));
    let tick = Stroke::new(1.0, axis.gamma_multiply(0.8));
    for d in [-1.0f32, 1.0] {
        let x = centre.x + d * radius;
        painter.line_segment([pos2(x, centre.y - 5.0), pos2(x, centre.y + 5.0)], tick);
        let y = centre.y + d * radius;
        painter.line_segment([pos2(centre.x - 5.0, y), pos2(centre.x + 5.0, y)], tick);
    }

    let mut mesh = egui::Mesh::default();
    let side = (half / 150.0).clamp(1.5, 3.0);
    let edge = REACH - side / radius;
    let at = |re: f32, im: f32| pos2(centre.x + re.clamp(-edge, edge) * radius, centre.y - im.clamp(-edge, edge) * radius);
    // Faint, so that the symbols landing on a point build up into it.
    let colour = Color32::from_rgba_unmultiplied(120, 220, 160, if points.len() > 1024 { 90 } else { 150 });
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_short_text_comes_back_whole() {
        assert_eq!(last_chars("N0CALL", 16), "N0CALL");
        assert_eq!(last_chars("de N0CALL N0CALL", 16), "de N0CALL N0CALL");
    }

    #[test]
    fn a_long_text_comes_back_as_its_last_sixteen() {
        assert_eq!(last_chars("CQ CQ CQ de N0CALL N0CALL", 16), "de N0CALL N0CALL");
    }

    #[test]
    fn characters_of_more_than_one_byte_are_kept_whole() {
        // Two bytes a character, as the listener makes of bytes from 0x80 up,
        // and then three and four. In each text a character straddles the
        // point sixteen bytes from the end, where a slice by bytes would panic.
        for (text, last) in [("Grüße aus Köln! 73", "üße aus Köln! 73"), ("1€2𝄞3€4𝄞5€6𝄞7€8𝄞9€", "2𝄞3€4𝄞5€6𝄞7€8𝄞9€")] {
            assert!(!text.is_char_boundary(text.len() - 16));
            assert_eq!(last_chars(text, 16), last);
        }
    }

    #[test]
    fn an_empty_text_comes_back_empty() {
        assert_eq!(last_chars("", 16), "");
        // As does any text when no characters are asked for.
        assert_eq!(last_chars("N0CALL", 0), "");
    }

    /// Make `edit` to `settings`, as a click in the row would, and say what
    /// open lines are then told: the commands in their debug form, which is
    /// plainer to read and to compare than a match on each.
    fn sent(settings: &mut Settings, edit: impl FnOnce(&mut Settings)) -> String {
        let before = settings.clone();
        edit(settings);
        format!("{:?}", switched(&before, settings))
    }

    #[test]
    fn a_change_reopens_only_the_line_it_belongs_to() {
        // As the window first runs: on the loopback, with nothing chosen.
        let mut s = Settings::default();
        // Off the loopback, with neither of the radio's devices chosen yet:
        // the engine is told, and waits for them.
        assert_eq!(sent(&mut s, |s| s.loopback = false), r#"[Radio(Some(("", "")))]"#);
        assert_eq!(sent(&mut s, |s| s.radio_in = "A".into()), r#"[Radio(Some(("A", "")))]"#);
        assert_eq!(sent(&mut s, |s| s.radio_out = "B".into()), r#"[Radio(Some(("A", "B")))]"#);
        // A mic with no speaker leaves the operator's line as it was, on
        // neither; the speaker too opens it.
        assert_eq!(sent(&mut s, |s| s.mic = "M".into()), "[]");
        assert_eq!(sent(&mut s, |s| s.speaker = "S".into()), r#"[Operator(Some(("M", "S")))]"#);
        assert_eq!(sent(&mut s, |s| s.mic = "N".into()), r#"[Operator(Some(("N", "S")))]"#);
        assert_eq!(sent(&mut s, |s| s.loopback = true), "[Radio(None)]");
        // The loopback's noise has its own command, and the rest of the
        // settings nothing to do with the lines.
        assert_eq!(sent(&mut s, |s| s.loopback_snr = 20.0), "[]");
        assert_eq!(sent(&mut s, |s| s.level = -20.0), "[]");
    }
}
