//! The engine: a thread that runs the radio and the operator's audio in
//! real time, and tells the window what is happening.
//!
//! Two sound-card lines, each a [`line::Duplex`] at the modem's 16 kHz:
//!
//! - the radio's: its receive audio in, its transmit audio out;
//! - the operator's: the microphone in, the speaker out.
//!
//! Or, with no radio at all, a loopback: the transmitter's samples go
//! straight to the receiver with noise added, so a station can hear what the
//! modem and the codec do to speech before going on the air.
//!
//! Every ten milliseconds the engine takes whatever each line has delivered
//! and tops up whatever each has waiting to go out: received audio into the
//! receiver, whose speech goes into a playout buffer for the speaker; the
//! microphone into the codec while the transmitter is keyed, and symbols out
//! to the radio as its line drains.

use std::collections::VecDeque;
use std::sync::mpsc::{Receiver as Inbox, Sender};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use dsp::Spectrum;
use line::Duplex;
use modem::channel::Random;
use modem::profile::{FS, PILOT, PREAMBLE, SLOT};
use modem::{Event, Modulation, Modulator, Receiver, State, VoiceMode, VoiceTx};
use voice::Playout;

use crate::speech::{Heard, ListenerThread, Talked, TalkerThread};

/// How often the engine wakes.
const TICK: Duration = Duration::from_millis(10);

/// Line samples kept waiting for each sound card: enough to ride out a late
/// tick, little enough not to add delay anyone notices -- 100 ms for the
/// radio and 60 ms for the speaker. With the codecs off this thread the loop
/// is never late by more than a tick or two.
const RADIO_AHEAD: usize = 1600;
const SPEAKER_AHEAD: usize = 960;

/// Symbols the modulator is left with before the next codeword is built:
/// half a slot, so a codeword carries speech right up to the moment it has to
/// go, and still four ticks' warning at 1600 baud.
const SYMBOLS_AHEAD: usize = SLOT / 2;

/// Codewords' worth of speech a live talker may be behind before the oldest
/// of it is dropped, silence or not.
const BEHIND_CODEWORDS: usize = 3;

/// Log lines kept.
const LOG: usize = 200;

/// What the window can ask for.
#[derive(Debug, Clone)]
pub enum Command {
    /// Open the lines: the radio's two devices, or a loopback at `snr_db`
    /// if `radio` is None; and the operator's microphone and speaker, if
    /// named.
    Open { radio: Option<(String, String)>, loopback_snr: f64, operator: Option<(String, String)> },
    /// Reopen the radio's line alone: on its two devices, or on the loopback
    /// at the SNR last given if None. A device's name left empty means it
    /// has not been chosen yet, and the radio waits closed for both.
    Radio(Option<(String, String)>),
    /// Reopen the operator's line alone: on the microphone and speaker
    /// named, or on neither if None.
    Operator(Option<(String, String)>),
    Close,
    /// Speech, at the modem's rate, to transmit in place of the microphone
    /// while keyed: a recording, or the self test's synthetic voice.
    Speak(Vec<f32>),
    /// Keep the speech played out, from now.
    Record(bool),
    /// Hand over the speech kept, and stop keeping it.
    TakeRecording(Sender<Vec<f32>>),
    Mode(&'static VoiceMode),
    Text(String),
    Level(f64),
    /// The loopback's noise, as Es/N0 in decibels, from now.
    LoopbackSnr(f64),
    Ptt(bool),
    /// Keep decoding while transmitting: a satellite's own downlink.
    FullDuplex(bool),
    Quit,
}

/// What the window shows.
#[derive(Debug, Clone, Default)]
pub struct Status {
    /// The lines that are open, or why they are not.
    pub lines: String,
    pub open: bool,
    pub transmitting: bool,
    /// Seconds of speech and symbols still to go out.
    pub behind: f64,
    pub receiving: String,
    pub rx_mode: Option<&'static VoiceMode>,
    pub snr_db: Option<f64>,
    pub offset_hz: Option<f64>,
    pub drift_ppm: Option<f64>,
    pub rx_level_dbfs: f64,
    pub mic_level_dbfs: f64,
    pub heard: usize,
    pub lost: usize,
    pub text: String,
    /// The receiver's recent points, in the modulation they were sent in,
    /// and the signal to noise of the burst they came from.
    pub points: Vec<[f32; 2]>,
    pub modulation: Option<Modulation>,
    pub scope_snr_db: Option<f64>,
    pub spectrum: Vec<f32>,
    pub hz_per_bin: f64,
    pub log: VecDeque<String>,
}

/// The radio when there is no radio: the transmitter's samples to the
/// receiver, with noise, at the pace of the clock on the wall.
#[derive(Debug)]
struct Loopback {
    snr_db: f64,
    random: Random,
    air: VecDeque<f32>,
    started: Instant,
    delivered: u64,
}

impl Loopback {
    fn new(snr_db: f64) -> Self {
        Self { snr_db, random: Random::new(7), air: VecDeque::new(), started: Instant::now(), delivered: 0 }
    }

    fn receive(&mut self, into: &mut Vec<f32>, level_dbfs: f64, baud: f64) {
        let power = 10f64.powf(level_dbfs / 10.0) / 2.0;
        let sigma = (FS / 2.0 / baud * power * 10f64.powf(-self.snr_db / 10.0)).sqrt();
        let due = (self.started.elapsed().as_secs_f64() * FS) as u64;
        while self.delivered < due {
            let x = f64::from(self.air.pop_front().unwrap_or(0.0)) + sigma * self.random.gaussian();
            into.push(x as f32);
            self.delivered += 1;
        }
    }
}

struct Engine {
    radio: Option<Duplex>,
    loopback: Option<Loopback>,
    operator: Option<Duplex>,
    receiver: Receiver,
    /// The codecs, on threads of their own; the mode last heard, and the
    /// text its transmission carries.
    listener: ListenerThread,
    talker: TalkerThread,
    rx_mode: Option<&'static VoiceMode>,
    rx_text: String,
    playout: Playout,
    modulator: Modulator,
    tx: Option<VoiceTx>,
    /// The talker has been told to finish and has not yet said it has.
    finishing: bool,
    mode: &'static VoiceMode,
    text: String,
    level: f64,
    ptt: bool,
    /// Keyed again while the last transmission was still going out.
    rekeyed: bool,
    full_duplex: bool,
    spectrum: Spectrum,
    status: Arc<Mutex<Status>>,
    log: VecDeque<String>,
    heard: usize,
    lost: usize,
    rx_level: f64,
    mic_level: f64,
    /// What each line is open on, or why it is not; None while it is closed.
    /// Kept apart, so that reopening one line leaves what the other says.
    radio_line: Option<String>,
    operator_line: Option<String>,
    /// The loopback's noise, as Es/N0 in decibels, kept while the radio is
    /// on its devices so that the loopback comes back as it was.
    loopback_snr: f64,
    started: Instant,
    /// Speech to send in place of the microphone, and the speech played
    /// out, kept while recording.
    script: VecDeque<f32>,
    recording: Option<Vec<f32>>,
    /// The last signal to noise a burst measured, kept for the scope.
    scope_snr: Option<f64>,
    /// Times the speech has run dry, as last said.
    starved: u64,
}

/// Start the engine; the handle to talk to it, and where it reports.
pub fn spawn(mode: &'static VoiceMode) -> (Sender<Command>, Arc<Mutex<Status>>) {
    let (commands, inbox) = std::sync::mpsc::channel();
    let status = Arc::new(Mutex::new(Status { lines: "no lines open".into(), ..Status::default() }));
    let shared = Arc::clone(&status);
    std::thread::Builder::new()
        .name("engine".into())
        .spawn(move || run(inbox, shared, mode))
        .expect("the engine thread starts");
    (commands, status)
}

fn run(inbox: Inbox<Command>, status: Arc<Mutex<Status>>, mode: &'static VoiceMode) {
    let mut engine = Engine::new(mode, status);
    let mut published = Instant::now();
    loop {
        while let Ok(command) = inbox.try_recv() {
            if matches!(command, Command::Quit) {
                return;
            }
            engine.command(command);
        }
        engine.tick();
        if published.elapsed() >= Duration::from_millis(40) {
            engine.publish();
            published = Instant::now();
        }
        std::thread::sleep(TICK);
    }
}

impl Engine {
    /// An engine with no lines open, its transmitter in `mode` until told
    /// otherwise, reporting to `status`.
    fn new(mode: &'static VoiceMode, status: Arc<Mutex<Status>>) -> Self {
        let mut engine = Self {
            radio: None,
            loopback: None,
            operator: None,
            receiver: Receiver::default(),
            listener: ListenerThread::spawn(),
            talker: TalkerThread::spawn(),
            rx_mode: None,
            rx_text: String::new(),
            playout: Playout::new(0),
            modulator: Modulator::new(mode.profile, -12.0),
            tx: None,
            finishing: false,
            mode,
            text: String::new(),
            level: -12.0,
            ptt: false,
            rekeyed: false,
            full_duplex: false,
            spectrum: Spectrum::new(1024, FS),
            status,
            log: VecDeque::new(),
            heard: 0,
            lost: 0,
            rx_level: -120.0,
            mic_level: -120.0,
            radio_line: None,
            operator_line: None,
            loopback_snr: 12.0,
            started: Instant::now(),
            script: VecDeque::new(),
            recording: None,
            scope_snr: None,
            starved: 0,
        };
        engine.set_preroll();
        engine
    }

    fn say(&mut self, line: impl Into<String>) {
        let t = self.started.elapsed().as_secs_f64();
        self.log.push_back(format!("{:>7.1}  {}", t, line.into()));
        while self.log.len() > LOG {
            self.log.pop_front();
        }
    }

    /// Enough speech gathered before playing to ride out the longest wait
    /// for more: a codeword's air time, and the next burst's preamble on top
    /// of it, which is air time that carries no speech. With one codeword's
    /// worth, the speech ran dry at every preamble.
    fn set_preroll(&mut self) {
        let mode = self.rx_mode.unwrap_or(self.mode);
        let seconds = mode.codeword_seconds() + mode.profile.seconds(PREAMBLE + PILOT) + 0.1;
        self.playout = Playout::new((seconds * FS) as usize);
        self.starved = 0;
    }

    fn command(&mut self, command: Command) {
        match command {
            Command::Open { radio, loopback_snr, operator } => self.open(radio, loopback_snr, operator),
            Command::Radio(radio) => {
                self.open_radio(radio);
                let said = self.radio_line.clone().unwrap_or_default();
                self.say(said);
            }
            Command::Operator(operator) => {
                self.open_operator(operator);
                let said = self.operator_line.clone().unwrap_or_default();
                self.say(said);
            }
            Command::Speak(speech) => self.script.extend(speech),
            Command::Record(on) => self.recording = on.then(Vec::new),
            Command::TakeRecording(reply) => {
                let _ = reply.send(self.recording.take().unwrap_or_default());
            }
            Command::Close => {
                self.radio = None;
                self.loopback = None;
                self.operator = None;
                self.radio_line = None;
                self.operator_line = None;
                self.tx = None;
                self.modulator.clear();
                self.say("lines closed");
            }
            Command::Mode(mode) => {
                self.mode = mode;
                self.say(format!("transmit mode {}", mode.name));
                // Open the codec now, on the listener's thread, so that a
                // neural codec's weights are loaded before the first
                // transmission rather than during it, and a codec that cannot
                // be used says so at once.
                self.listener.warm(mode.codec);
            }
            Command::Text(text) => self.text = text,
            Command::Level(level) => {
                self.level = level;
                self.modulator.set_level(level);
            }
            Command::Ptt(down) => self.key(down),
            Command::FullDuplex(on) => self.full_duplex = on,
            Command::LoopbackSnr(snr) => {
                self.loopback_snr = snr;
                if let Some(lb) = &mut self.loopback {
                    lb.snr_db = snr;
                    self.radio_line = Some(format!("loopback at {snr:.0} dB"));
                }
            }
            Command::Quit => {}
        }
    }

    fn open(&mut self, radio: Option<(String, String)>, snr: f64, operator: Option<(String, String)>) {
        // Both old lines close before either new one opens: the new radio
        // may want a device the old operator's line still holds.
        self.radio = None;
        self.loopback = None;
        self.operator = None;
        self.loopback_snr = snr;
        self.open_radio(radio);
        self.open_operator(operator);
        let lines = self.lines();
        self.say(lines);
    }

    /// Put the radio's line on `radio`'s devices, or on the loopback if None,
    /// in place of whatever it was on. Nothing else stops: the operator's
    /// line runs on, a transmission going out carries on into the new line,
    /// and the receiver and the playout keep what they have.
    fn open_radio(&mut self, radio: Option<(String, String)>) {
        // The old line stops before the new one starts, since a sound card
        // may not open a device again while a stream still holds it.
        self.radio = None;
        self.loopback = None;
        let said = match radio {
            None => {
                self.loopback = Some(Loopback::new(self.loopback_snr));
                format!("loopback at {:.0} dB", self.loopback_snr)
            }
            // Not chosen is not a failure, and no reason to guess: the
            // default device is as likely the operator's own headset.
            Some((input, output)) if input.is_empty() || output.is_empty() => {
                "radio waiting for both devices to be chosen".to_string()
            }
            Some((input, output)) => match Duplex::open(Some(&input), Some(&output), FS) {
                Ok(d) => {
                    let said = format!("radio {} / {}", d.input_device, d.output_device);
                    self.radio = Some(d);
                    said
                }
                Err(e) => format!("radio would not open: {e}"),
            },
        };
        self.radio_line = Some(said);
    }

    /// Put the operator's line on `operator`'s microphone and speaker, or on
    /// neither if None, in place of whatever it was on; the rest runs on, as
    /// for the radio's.
    fn open_operator(&mut self, operator: Option<(String, String)>) {
        // Stopped first, as the radio's is.
        self.operator = None;
        let said = match operator.map(|(mic, speaker)| Duplex::open(Some(&mic), Some(&speaker), FS)) {
            Some(Ok(d)) => {
                let said = format!("mic {} / speaker {}", d.input_device, d.output_device);
                self.operator = Some(d);
                said
            }
            Some(Err(e)) => format!("mic and speaker would not open: {e}"),
            None => "no mic or speaker".into(),
        };
        self.operator_line = Some(said);
    }

    /// The lines, as the window shows them.
    fn lines(&self) -> String {
        match (&self.radio_line, &self.operator_line) {
            (Some(radio), Some(operator)) => format!("{radio}; {operator}"),
            (Some(one), None) | (None, Some(one)) => one.clone(),
            (None, None) => "no lines open".into(),
        }
    }

    fn key(&mut self, down: bool) {
        if down == self.ptt {
            return;
        }
        self.ptt = down;
        if !down {
            self.rekeyed = false;
            if self.tx.is_some() && !self.finishing {
                // The last words, still in the codec, go out before the end:
                // the transmission ends when the talker says it has handed
                // them all back.
                self.talker.finish();
                self.finishing = true;
            }
            return;
        }
        if self.tx.is_some() {
            // The last transmission is still going out; this one follows it.
            self.rekeyed = true;
            return;
        }
        self.start_transmission();
    }

    fn start_transmission(&mut self) {
        self.talker.start(self.mode);
        self.modulator.set_profile(self.mode.profile);
        self.modulator.set_level(self.level);
        self.tx = Some(
            VoiceTx::new(self.mode, crate::cli::stream_id(), &self.text)
                .with_cap(BEHIND_CODEWORDS * self.mode.frames_per_codeword()),
        );
        self.finishing = false;
        self.say(format!("transmitting {}", self.mode.name));
    }

    fn tick(&mut self) {
        // The radio's receive audio, or the loopback's.
        let mut input = Vec::new();
        if let Some(radio) = &self.radio {
            radio.receive(&mut input);
        } else if let Some(lb) = &mut self.loopback {
            lb.receive(&mut input, self.level, self.mode.profile.baud());
        }
        let deaf = self.tx.is_some() && !self.full_duplex && self.loopback.is_none();
        let mut power = 0.0;
        for &x in &input {
            self.spectrum.push(f64::from(x));
            power += f64::from(x) * f64::from(x);
            self.receiver.feed(if deaf { 0.0 } else { x });
        }
        if !input.is_empty() {
            let db = 10.0 * (2.0 * power / input.len() as f64).max(1e-12).log10();
            self.rx_level += 0.3 * (db - self.rx_level);
        }
        while let Some(event) = self.receiver.event() {
            self.on_event(event);
        }

        // The microphone, into the codec while keyed; or the script, as
        // fast as the radio's line runs, in its place.
        let mut mic = Vec::new();
        if let Some(op) = &self.operator {
            op.receive(&mut mic);
        }
        if !self.script.is_empty() {
            let n = input.len().min(self.script.len());
            mic = self.script.drain(..n).collect();
        }
        if !mic.is_empty() {
            let power = mic.iter().map(|x| f64::from(*x).powi(2)).sum::<f64>() / mic.len() as f64;
            let db = 10.0 * (2.0 * power).max(1e-12).log10();
            self.mic_level += 0.3 * (db - self.mic_level);
        }
        if self.ptt && self.tx.is_some() && !self.finishing {
            self.talker.speak(mic);
        }
        // Whatever the codecs have finished since the last tick.
        for done in self.talker.done() {
            match done {
                Talked::Frames(frames) => {
                    if let Some(tx) = &mut self.tx {
                        for frame in frames {
                            tx.push_spoken(frame.bits, frame.quiet);
                        }
                    }
                }
                Talked::Finished => {
                    self.finishing = false;
                    if let Some(tx) = &mut self.tx {
                        tx.end();
                    }
                }
                Talked::Failed(e) => self.say(format!("cannot transmit: {e}")),
            }
        }
        for done in self.listener.done() {
            match done {
                Heard::Speech(speech, text) => {
                    self.playout.push(&speech);
                    self.rx_text = text;
                }
                Heard::Failed(e) => self.say(e),
            }
        }

        // Symbols to the radio as its line drains.
        if let Some(tx) = &mut self.tx {
            while self.modulator.queued() < SYMBOLS_AHEAD {
                match tx.next_symbols() {
                    Some(symbols) => self.modulator.push(&symbols),
                    None => break,
                }
            }
        }
        let waiting = match (&self.radio, &self.loopback) {
            (Some(r), _) => r.pending(),
            (None, Some(lb)) => lb.air.len(),
            (None, None) => usize::MAX,
        };
        if waiting < RADIO_AHEAD {
            let mut out = Vec::with_capacity(RADIO_AHEAD - waiting);
            self.modulator.fill(RADIO_AHEAD - waiting, &mut out);
            if let Some(r) = &self.radio {
                r.transmit(&out);
            } else if let Some(lb) = &mut self.loopback {
                lb.air.extend(out);
            }
        }
        if self.tx.as_ref().is_some_and(VoiceTx::is_done) && !self.modulator.busy() {
            self.tx = None;
            self.say("transmission over");
            if self.rekeyed && self.ptt {
                self.rekeyed = false;
                self.start_transmission();
            }
        }

        // Speech to the speaker; or, with none, played out to nothing at the
        // radio line's pace, so that a recording still hears it in time.
        let mut out = Vec::new();
        match &self.operator {
            Some(op) => {
                let waiting = op.pending();
                if waiting < SPEAKER_AHEAD {
                    self.playout.pull(SPEAKER_AHEAD - waiting, &mut out);
                    op.transmit(&out);
                }
            }
            None => self.playout.pull(input.len(), &mut out),
        }
        if self.playout.starved > self.starved {
            self.starved = self.playout.starved;
            self.say("the speech ran dry and waited for more");
        }
        if let Some(recording) = &mut self.recording {
            recording.extend_from_slice(&out);
        }
    }

    fn on_event(&mut self, event: Event) {
        match event {
            Event::Heard { profile, header, snr_db, offset_hz, .. } => {
                let what = VoiceMode::of(profile, &header).map_or_else(|| format!("{:?} burst", header.kind), |m| m.name.to_string());
                self.say(format!("heard {what}: {snr_db:.1} dB, {offset_hz:+.1} Hz"));
            }
            Event::Untrained { .. } => self.say("a preamble, but the equaliser would not train on it"),
            Event::Voice { mode, codeword, .. } => {
                if self.rx_mode != Some(mode) {
                    self.rx_mode = Some(mode);
                    self.set_preroll();
                }
                if codeword.is_some() {
                    self.heard += 1;
                } else {
                    self.lost += 1;
                }
                self.listener.hear(mode, codeword);
            }
            Event::BurstEnd(r) => {
                self.say(format!(
                    "burst over: {} heard, {} lost, {:.1} dB, {:+.1} Hz, {:+.0} ppm{}",
                    r.ok,
                    r.failed,
                    r.snr_db,
                    r.offset_hz,
                    r.drift_ppm,
                    if r.aborted { ", faded out" } else { "" }
                ));
            }
            Event::Delivered(Ok(delivery)) => self.say(format!("received transfer {}: {:?}", delivery.transfer, delivery.content)),
            Event::Delivered(Err(e)) => self.say(format!("a transfer arrived damaged: {e}")),
            Event::Slip { by } => self.say(format!("the audio slipped {by} symbols")),
            Event::Trained { .. } | Event::Block { .. } => {}
        }
    }

    fn publish(&mut self) {
        if let Some(snr) = self.receiver.snr_db() {
            self.scope_snr = Some(snr);
        }
        let receiving = match self.receiver.state() {
            State::Listening => "listening".to_string(),
            State::Training(p, _) => format!("training on a {} preamble", p.name()),
            State::Receiving { profile, header, done } => {
                let mode = VoiceMode::of(profile, &header).map_or("data", |m| m.name);
                format!("receiving {mode}, codeword {} of {}", done + 1, header.codewords)
            }
        };
        let mut spectrum = vec![0.0; self.spectrum.size() / 2];
        let mut bins = vec![0.0f64; self.spectrum.size() / 2];
        if self.spectrum.ready() {
            self.spectrum.magnitudes_db(&mut bins);
            for (s, b) in spectrum.iter_mut().zip(&bins) {
                *s = *b as f32;
            }
        }
        let points: Vec<[f32; 2]> = self.receiver.points().iter().map(|z| [z.re as f32, z.im as f32]).collect();
        let behind = self.tx.as_ref().map_or(0.0, |tx| {
            tx.waiting() as f64 * self.mode.codec.frame_seconds() + self.mode.profile.seconds(self.modulator.queued())
        });
        let Ok(mut status) = self.status.lock() else { return };
        *status = Status {
            lines: self.lines(),
            open: self.operator.is_some() || self.radio.is_some() || self.loopback.is_some(),
            transmitting: self.tx.is_some(),
            behind,
            receiving,
            rx_mode: self.rx_mode,
            snr_db: self.receiver.snr_db(),
            offset_hz: self.receiver.offset_hz(),
            drift_ppm: self.receiver.drift_ppm(),
            rx_level_dbfs: self.rx_level,
            mic_level_dbfs: self.mic_level,
            heard: self.heard,
            lost: self.lost,
            text: self.rx_text.clone(),
            modulation: self.receiver.modulation(),
            points,
            scope_snr_db: self.scope_snr,
            spectrum,
            hz_per_bin: FS / self.spectrum.size() as f64,
            log: self.log.clone(),
        };
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use modem::Profile;

    /// A device no machine has. Asking for it fails in the lookup, before
    /// anything is opened: a test must never open a real sound card, and the
    /// machines tests run on mostly have none.
    const NOWHERE: &str = "no such device: a voicemodem test";

    /// A line's two devices, both [`NOWHERE`].
    fn nowhere() -> Option<(String, String)> {
        Some((NOWHERE.into(), NOWHERE.into()))
    }

    /// An engine on the loopback at 15 dB, and where it reports. Its
    /// operator's line is asked for on devices that do not exist, so that it
    /// has something of its own to say.
    fn on_the_loopback() -> (Engine, Arc<Mutex<Status>>) {
        let status = Arc::new(Mutex::new(Status::default()));
        let mut engine = Engine::new(VoiceMode::robust(Profile::Narrow), Arc::clone(&status));
        engine.command(Command::Open { radio: None, loopback_snr: 15.0, operator: nowhere() });
        (engine, status)
    }

    /// The engine's loop without its thread: a tick every [`TICK`] until
    /// `done` says so or `seconds` have gone; whether it did.
    fn tick_until(engine: &mut Engine, seconds: f64, done: impl Fn(&Engine) -> bool) -> bool {
        let until = Instant::now() + Duration::from_secs_f64(seconds);
        while !done(engine) {
            if Instant::now() >= until {
                return false;
            }
            std::thread::sleep(TICK);
            engine.tick();
        }
        true
    }

    #[test]
    fn the_radio_switches_alone_and_comes_back_to_the_loopback() {
        let (mut engine, status) = on_the_loopback();
        let operator = engine.operator_line.clone();
        assert!(operator.as_deref().is_some_and(|o| o.starts_with("mic and speaker would not open")), "{operator:?}");

        engine.command(Command::Radio(nowhere()));
        let radio = engine.radio_line.clone().unwrap_or_default();
        assert!(radio.starts_with("radio would not open"), "{radio}");
        assert_eq!(engine.operator_line, operator, "the operator's line changed with the radio's");
        // A while with the radio's line down, and nothing arrives.
        tick_until(&mut engine, 0.05, |_| false);
        assert!(engine.rx_level <= -120.0, "something arrived with no line for it to arrive on");

        engine.command(Command::Radio(None));
        assert_eq!(engine.radio_line.as_deref(), Some("loopback at 15 dB"));
        assert_eq!(engine.operator_line, operator);
        assert!(tick_until(&mut engine, 1.0, |e| e.rx_level > -120.0), "nothing arrived from the loopback");
        engine.publish();
        let view = status.lock().map(|s| s.clone()).unwrap_or_default();
        assert!(view.open);
        assert_eq!(view.lines, format!("loopback at 15 dB; {}", operator.unwrap_or_default()));

        engine.command(Command::Close);
        assert_eq!(engine.lines(), "no lines open");
    }

    #[test]
    fn the_operator_switches_alone_and_the_loopback_runs_on() {
        let (mut engine, _) = on_the_loopback();
        let delivered = |e: &Engine| e.loopback.as_ref().map_or(0, |lb| lb.delivered);
        assert!(tick_until(&mut engine, 1.0, |e| delivered(e) > 0), "nothing arrived from the loopback");
        engine.command(Command::Operator(None));
        assert_eq!(engine.operator_line.as_deref(), Some("no mic or speaker"));
        assert_eq!(engine.radio_line.as_deref(), Some("loopback at 15 dB"));
        // A loopback opened afresh would have delivered nothing yet.
        assert!(delivered(&engine) > 0, "the loopback was reopened with the operator's line");
    }

    #[test]
    fn a_radio_half_chosen_waits_and_the_loopback_keeps_its_noise() {
        let (mut engine, _) = on_the_loopback();
        engine.command(Command::LoopbackSnr(25.0));
        assert_eq!(engine.radio_line.as_deref(), Some("loopback at 25 dB"));
        // One device chosen and the other not yet: nothing is opened, and
        // nothing guessed at.
        engine.command(Command::Radio(Some((String::new(), NOWHERE.into()))));
        assert!(engine.radio.is_none() && engine.loopback.is_none());
        assert_eq!(engine.radio_line.as_deref(), Some("radio waiting for both devices to be chosen"));
        // Back on the loopback, at the noise it was left at.
        engine.command(Command::Radio(None));
        assert_eq!(engine.radio_line.as_deref(), Some("loopback at 25 dB"));
        assert!(engine.loopback.as_ref().is_some_and(|lb| (lb.snr_db - 25.0).abs() < 1e-9));
    }

    #[test]
    fn a_transmission_carries_on_across_a_switch() {
        let (mut engine, _) = on_the_loopback();
        engine.command(Command::Ptt(true));
        assert!(tick_until(&mut engine, 1.0, |e| e.modulator.busy()), "the transmission never began");
        engine.playout.push(&[0.25; 480]);
        let playing = engine.playout.len();

        engine.command(Command::Radio(nowhere()));
        engine.command(Command::Operator(None));
        engine.command(Command::Radio(None));
        assert!(engine.ptt && engine.tx.is_some() && engine.modulator.busy(), "the transmission was dropped");
        assert_eq!(engine.playout.len(), playing, "the speech waiting to be played was dropped");

        // It goes on into the loopback it was switched back to, and ends there.
        engine.command(Command::Ptt(false));
        assert!(tick_until(&mut engine, 10.0, |e| e.tx.is_none()), "the transmission never ended: {:?}", engine.log);
    }

    /// The loopback's noise power over the next `seconds`, with nothing on
    /// the air.
    fn noise_power(lb: &mut Loopback, seconds: f64) -> f64 {
        let mut got = Vec::new();
        let until = Instant::now() + Duration::from_secs_f64(seconds);
        while Instant::now() < until {
            std::thread::sleep(Duration::from_millis(5));
            lb.receive(&mut got, -12.0, 1600.0);
        }
        got.iter().map(|x| f64::from(*x).powi(2)).sum::<f64>() / got.len().max(1) as f64
    }

    #[test]
    fn the_loopback_takes_a_new_snr_while_it_runs() {
        let mut lb = Loopback::new(10.0);
        let before = noise_power(&mut lb, 0.1);
        // What the window's slider now sends, straight into the running
        // loopback.
        lb.snr_db = 30.0;
        let after = noise_power(&mut lb, 0.1);
        let fell = 10.0 * (before / after).log10();
        assert!((fell - 20.0).abs() < 1.0, "noise fell {fell:.1} dB for a 20 dB step");
    }
}
