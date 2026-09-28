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
use modem::profile::{FS, SLOT};
use modem::{Event, Modulator, Receiver, State, VoiceMode, VoiceTx};
use voice::{Listener, Playout, Talker};

/// How often the engine wakes.
const TICK: Duration = Duration::from_millis(10);

/// Line samples kept waiting for each sound card: enough to ride out a late
/// tick, little enough not to add delay anyone notices.
const RADIO_AHEAD: usize = 2400;
const SPEAKER_AHEAD: usize = 1600;

/// Symbols the modulator is left with before the next codeword is built:
/// a slot's worth, so a codeword carries speech right up to the moment it
/// has to go.
const SYMBOLS_AHEAD: usize = SLOT + 16;

/// Log lines kept.
const LOG: usize = 200;

/// What the window can ask for.
#[derive(Debug, Clone)]
pub enum Command {
    /// Open the lines: the radio's two devices, or a loopback at `snr_db`
    /// if `radio` is None; and the operator's microphone and speaker, if
    /// named.
    Open { radio: Option<(String, String)>, loopback_snr: f64, operator: Option<(String, String)> },
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
    pub points: Vec<[f32; 2]>,
    pub modulation_points: usize,
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
    listener: Option<Listener>,
    playout: Playout,
    modulator: Modulator,
    tx: Option<VoiceTx>,
    talker: Option<Talker>,
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
    lines: String,
    started: Instant,
    /// Speech to send in place of the microphone, and the speech played
    /// out, kept while recording.
    script: VecDeque<f32>,
    recording: Option<Vec<f32>>,
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
    let mut engine = Engine {
        radio: None,
        loopback: None,
        operator: None,
        receiver: Receiver::default(),
        listener: None,
        playout: Playout::new(0),
        modulator: Modulator::new(mode.profile, -12.0),
        tx: None,
        talker: None,
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
        lines: "no lines open".into(),
        started: Instant::now(),
        script: VecDeque::new(),
        recording: None,
    };
    engine.set_preroll();
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
    fn say(&mut self, line: impl Into<String>) {
        let t = self.started.elapsed().as_secs_f64();
        self.log.push_back(format!("{:>7.1}  {}", t, line.into()));
        while self.log.len() > LOG {
            self.log.pop_front();
        }
    }

    /// Enough speech gathered before playing to ride out a codeword's wait.
    fn set_preroll(&mut self) {
        let seconds = self.listener.as_ref().map_or(self.mode, |l| l.mode()).codeword_seconds();
        self.playout = Playout::new((seconds * FS) as usize);
    }

    fn command(&mut self, command: Command) {
        match command {
            Command::Open { radio, loopback_snr, operator } => self.open(radio, loopback_snr, operator),
            Command::Speak(speech) => self.script.extend(speech),
            Command::Record(on) => self.recording = on.then(Vec::new),
            Command::TakeRecording(reply) => {
                let _ = reply.send(self.recording.take().unwrap_or_default());
            }
            Command::Close => {
                self.radio = None;
                self.loopback = None;
                self.operator = None;
                self.tx = None;
                self.modulator.clear();
                self.lines = "no lines open".into();
                self.say("lines closed");
            }
            Command::Mode(mode) => {
                self.mode = mode;
                self.say(format!("transmit mode {}", mode.name));
                // Open the codec now, so that a neural codec's weights are
                // loaded before the first transmission rather than during it,
                // and a codec that cannot be used says so at once.
                if let Err(e) = voice::open(mode.codec) {
                    self.say(format!("{} cannot be used: {e}", mode.codec.label()));
                }
            }
            Command::Text(text) => self.text = text,
            Command::Level(level) => {
                self.level = level;
                self.modulator.set_level(level);
            }
            Command::Ptt(down) => self.key(down),
            Command::FullDuplex(on) => self.full_duplex = on,
            Command::Quit => {}
        }
    }

    fn open(&mut self, radio: Option<(String, String)>, snr: f64, operator: Option<(String, String)>) {
        self.radio = None;
        self.loopback = None;
        self.operator = None;
        let mut described = Vec::new();
        match radio {
            Some((input, output)) => match Duplex::open(Some(&input), Some(&output), FS) {
                Ok(d) => {
                    described.push(format!("radio {} / {}", d.input_device, d.output_device));
                    self.radio = Some(d);
                }
                Err(e) => described.push(format!("radio would not open: {e}")),
            },
            None => {
                described.push(format!("loopback at {snr:.0} dB"));
                self.loopback = Some(Loopback::new(snr));
            }
        }
        match operator.map(|(mic, speaker)| Duplex::open(Some(&mic), Some(&speaker), FS)) {
            Some(Ok(d)) => {
                described.push(format!("mic {} / speaker {}", d.input_device, d.output_device));
                self.operator = Some(d);
            }
            Some(Err(e)) => described.push(format!("mic and speaker would not open: {e}")),
            None => described.push("no mic or speaker".into()),
        }
        self.lines = described.join("; ");
        let lines = self.lines.clone();
        self.say(lines);
    }

    fn key(&mut self, down: bool) {
        if down == self.ptt {
            return;
        }
        self.ptt = down;
        if !down {
            self.rekeyed = false;
            if let Some(tx) = &mut self.tx {
                // The last words, still in the codec, go out before the end.
                if let Some(talker) = &mut self.talker {
                    let mut frames = Vec::new();
                    talker.finish(&mut frames);
                    for frame in frames {
                        tx.push_frame(frame);
                    }
                }
                tx.end();
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
        match Talker::new(self.mode, FS) {
            Ok(talker) => {
                self.talker = Some(talker);
                self.modulator.set_profile(self.mode.profile);
                self.modulator.set_level(self.level);
                self.tx = Some(VoiceTx::new(self.mode, crate::cli::stream_id(), &self.text));
                self.say(format!("transmitting {}", self.mode.name));
            }
            Err(e) => self.say(format!("cannot transmit: {e}")),
        }
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
        if self.ptt
            && let (Some(talker), Some(tx)) = (&mut self.talker, &mut self.tx)
        {
            let mut frames = Vec::new();
            talker.speak(&mic, &mut frames);
            for frame in frames {
                tx.push_frame(frame);
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
                if self.listener.as_ref().is_none_or(|l| l.mode() != mode) {
                    match Listener::new(mode, FS) {
                        Ok(l) => {
                            self.listener = Some(l);
                            self.set_preroll();
                        }
                        Err(e) => {
                            self.say(format!("cannot play {}: {e}", mode.name));
                            self.listener = None;
                        }
                    }
                }
                if codeword.is_some() {
                    self.heard += 1;
                } else {
                    self.lost += 1;
                }
                if let Some(listener) = &mut self.listener {
                    let mut speech = Vec::new();
                    listener.hear(codeword.as_ref(), &mut speech);
                    self.playout.push(&speech);
                }
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
            lines: self.lines.clone(),
            open: self.operator.is_some() || self.radio.is_some() || self.loopback.is_some(),
            transmitting: self.tx.is_some(),
            behind,
            receiving,
            rx_mode: self.listener.as_ref().map(Listener::mode),
            snr_db: self.receiver.snr_db(),
            offset_hz: self.receiver.offset_hz(),
            drift_ppm: self.receiver.drift_ppm(),
            rx_level_dbfs: self.rx_level,
            mic_level_dbfs: self.mic_level,
            heard: self.heard,
            lost: self.lost,
            text: self.listener.as_ref().map(Listener::text).unwrap_or_default(),
            modulation_points: self.receiver.modulation().map_or(0, |m| m.points()),
            points,
            spectrum,
            hz_per_bin: FS / self.spectrum.size() as f64,
            log: self.log.clone(),
        };
    }
}
