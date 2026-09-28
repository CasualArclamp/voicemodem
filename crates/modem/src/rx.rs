//! The receiver: line samples in, speech frames and transfers out.
//!
//! Detectors listen all the time, one for each profile and carrier offset
//! being listened for. When one finds a preamble and reads its header, a
//! fresh BinModem QAM core is made for the burst, mixing down with the
//! carrier the detector found, and the line is played into it again from
//! just before the preamble, so that it has every sample of it. The core is
//! told where the preamble's first symbol is, how far the carrier is still
//! off, and what all 300 symbols were; it solves its equaliser for them
//! outright by least squares -- the carrier's phase, the line's gain and its
//! delay distortion included -- and tracks from the first pilot on: timing,
//! carrier and equaliser following the far end, and a slip in timing or
//! phase found again from the raw samples. The framer takes the core's
//! points, finds the pilots, and hands back codewords.
//!
//! A new core for each burst, because each burst may come from a different
//! station, at a different level, through a different path. The core keeps
//! nothing from one to the next, and the detectors keep nothing at all.

use std::collections::VecDeque;

use dsp::Complex;
use dsp::qam::{Band, Core, Heard, Options, Training, Window};

use crate::detect::{Detector, Found};
use crate::frame::{Geometry, Header, Kind};
use crate::framer::{Framed, Framer, GIVE_UP_DATA, GIVE_UP_VOICE};
use crate::profile::{FS, PREAMBLE, Profile, SLOT, burst_symbols};
use crate::psk::Modulation;
use crate::transfer::{Assembler, Delivery};
use crate::voice::{VoiceCodeword, VoiceMode};

/// Line samples kept to play into a new core: long enough to reach back
/// over a whole preamble and the time it takes to find one, and more.
const HISTORY: usize = 4 * 16_000;

/// Samples before the preamble's first symbol that a new core is given:
/// its interpolating filter, the equaliser's reach and the training's
/// search all look that far back.
const LEAD: u64 = 400;

/// The equaliser is solved over the preamble from this symbol on, the first
/// few being where a far AGC or an FM squelch may still be opening.
const SOLVE_FROM: usize = 8;

/// Half symbols either side of where the detector put the preamble that
/// training searches.
const SEARCH: i64 = 8;

/// The carrier loop's frequency limit, either side of where the detector
/// put the carrier. BinModem's V.32 setting is 20 Hz, right for telephone
/// lines; a satellite's Doppler keeps moving after the preamble.
const TURN_LIMIT_HZ: f64 = 150.0;

/// The least signal to noise training may come to and still be tracked:
/// the header has passed its CRC, so the preamble is real, and the robust
/// modes work below BinModem's 6 dB.
const LEAST_TRAINED_DB: f64 = 1.0;

/// Samples a preamble found is held before it is acted on, for the rest of
/// a detector bank to find it too: three symbols narrow.
const HOLD: u64 = 30;

/// What the receiver has to say.
#[derive(Debug, Clone, PartialEq)]
pub enum Event {
    /// A preamble heard and its header read.
    Heard { profile: Profile, header: Header, snr_db: f64, offset_hz: f64, seconds: f64 },
    /// The equaliser trained on the preamble; the payload is being read.
    Trained { snr_db: f64 },
    /// The equaliser could not be trained: nothing of this burst is read.
    Untrained { header: Header },
    /// A data codeword decoded, or not.
    Block { header: Header, codeword: usize, block: Option<u16>, erased_slots: usize },
    /// A voice codeword: its frames if it decoded, None if it did not.
    Voice { mode: &'static VoiceMode, stream: u16, codeword: Option<VoiceCodeword>, erased_slots: usize },
    /// The symbols moved by a whole number of symbols.
    Slip { by: i64 },
    /// A burst is over.
    BurstEnd(BurstReport),
    /// A whole transfer arrived; or arrived and did not check.
    Delivered(Result<Delivery, String>),
}

/// How a burst went.
#[derive(Debug, Clone, PartialEq)]
pub struct BurstReport {
    pub profile: Profile,
    pub header: Header,
    pub ok: usize,
    pub failed: usize,
    pub slips: usize,
    /// Es/N0 as the pilots measured it at the end.
    pub snr_db: f64,
    /// Carrier offset from the profile's and far clock, as the core had
    /// them at the end.
    pub offset_hz: f64,
    pub drift_ppm: f64,
    /// Given up before the end, its pilots lost.
    pub aborted: bool,
}

/// What the receiver is doing, for a display.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum State {
    Listening,
    /// A preamble heard; waiting for the equaliser.
    Training(Profile, Header),
    /// Reading the payload: codewords done of the burst's.
    Receiving { profile: Profile, header: Header, done: usize },
}

#[derive(Debug, Clone)]
struct Active {
    found: Found,
    geometry: Geometry,
    /// The voice mode, for a voice burst.
    voice: Option<&'static VoiceMode>,
    core: Core,
    /// The next line sample to give the core.
    fed: u64,
    trained: bool,
    framer: Framer,
    ok: usize,
    failed: usize,
    slips: usize,
    /// Where the burst should end on the line, in samples.
    end: f64,
}

#[derive(Debug, Clone)]
pub struct Receiver {
    detectors: Vec<Detector>,
    raw: VecDeque<f32>,
    raw_first: u64,
    taken: u64,
    burst: Option<Box<Active>>,
    assembler: Assembler,
    events: VecDeque<Event>,
    /// The line's mean square, over about a tenth of a second.
    level: f64,
    /// Preambles heard while a burst was being read, and ignored.
    ignored: u64,
    last_points: Vec<Complex>,
    /// Preambles found and not yet acted on, and when to act on them.
    pending: Vec<Found>,
    pending_until: u64,
}

impl Default for Receiver {
    fn default() -> Self {
        Self::new(&Profile::ALL)
    }
}

impl Receiver {
    /// A receiver listening for `profiles`.
    pub fn new(profiles: &[Profile]) -> Self {
        let detectors = profiles
            .iter()
            .flat_map(|&p| p.search_offsets().iter().map(move |&o| Detector::new(p, o)))
            .collect();
        Self {
            detectors,
            raw: VecDeque::with_capacity(HISTORY),
            raw_first: 0,
            taken: 0,
            burst: None,
            assembler: Assembler::default(),
            events: VecDeque::new(),
            level: 0.0,
            ignored: 0,
            last_points: Vec::new(),
            pending: Vec::new(),
            pending_until: 0,
        }
    }

    /// The next event, if there is one.
    pub fn event(&mut self) -> Option<Event> {
        self.events.pop_front()
    }

    /// Samples taken so far.
    pub fn taken(&self) -> u64 {
        self.taken
    }

    /// The line's level, in dB relative to a full-scale sine.
    pub fn level_dbfs(&self) -> f64 {
        10.0 * (2.0 * self.level).max(1e-12).log10()
    }

    pub fn state(&self) -> State {
        match &self.burst {
            None => State::Listening,
            Some(a) if !a.trained => State::Training(a.found.profile, a.found.header),
            Some(a) => State::Receiving { profile: a.found.profile, header: a.found.header, done: a.ok + a.failed },
        }
    }

    /// Es/N0 of the burst being read, from its pilots.
    pub fn snr_db(&self) -> Option<f64> {
        self.burst.as_ref().filter(|a| a.trained).map(|a| a.framer.snr_db())
    }

    /// The far carrier's offset from the profile's, while a burst is being
    /// read.
    pub fn offset_hz(&self) -> Option<f64> {
        self.burst.as_ref().filter(|a| a.trained).map(|a| offset_of(a))
    }

    /// The far clock against ours, while a burst is being read.
    pub fn drift_ppm(&self) -> Option<f64> {
        self.burst.as_ref().filter(|a| a.trained).map(|a| a.core.drift_ppm())
    }

    /// The most recent data points, turned the right way round: the burst
    /// being read's, or the last one's.
    pub fn points(&self) -> Vec<Complex> {
        match &self.burst {
            Some(a) if a.trained => a.framer.points().copied().collect(),
            _ => self.last_points.clone(),
        }
    }

    /// The modulation of the burst being read.
    pub fn modulation(&self) -> Option<Modulation> {
        self.burst.as_ref().map(|a| a.geometry.modulation)
    }

    /// The blocks of transfer `id` still missing, if it is remembered.
    pub fn missing(&self, id: u16) -> Option<Vec<u16>> {
        self.assembler.missing(id)
    }

    pub fn progress(&self, id: u16) -> Option<(usize, usize)> {
        self.assembler.progress(id)
    }

    /// Preambles whose headers failed their CRC, and preambles ignored.
    pub fn rejected(&self) -> (u64, u64) {
        (self.detectors.iter().map(Detector::rejected).sum(), self.ignored)
    }

    /// Take one line sample.
    pub fn feed(&mut self, x: f32) {
        self.taken += 1;
        self.raw.push_back(x);
        if self.raw.len() > HISTORY {
            self.raw.pop_front();
            self.raw_first += 1;
        }
        self.level += (f64::from(x).powi(2) - self.level) / 1600.0;
        for detector in &mut self.detectors {
            detector.feed(f64::from(x));
            while let Some(f) = detector.found() {
                if self.pending.is_empty() {
                    self.pending_until = self.taken + HOLD;
                }
                self.pending.push(f);
            }
        }
        if !self.pending.is_empty() && self.taken >= self.pending_until {
            self.take_pending();
        }
        self.run_burst();
    }

    /// The preambles found in the last few symbols, best first, less those
    /// that are the same preamble as a better one: neighbouring offsets of a
    /// bank find the same preamble a few samples apart.
    fn take_pending(&mut self) {
        let mut found = std::mem::take(&mut self.pending);
        found.sort_by(|a, b| b.quality.total_cmp(&a.quality));
        let mut taken: Vec<f64> = Vec::new();
        for f in found {
            let same = taken.iter().any(|&start| (start - f.start).abs() < 4.0 * f.profile.sps());
            if !same {
                taken.push(f.start);
                self.on_found(f);
            }
        }
    }

    /// Take a run of samples.
    pub fn feed_all(&mut self, samples: &[f32]) {
        for &x in samples {
            self.feed(x);
        }
    }

    /// The line has ended: finish any burst with what it has.
    pub fn finish(&mut self) {
        if !self.pending.is_empty() {
            self.take_pending();
            self.run_burst();
        }
        if let Some(mut active) = self.burst.take() {
            if active.trained {
                active.framer.finish();
                self.framed(&mut active);
            } else {
                self.events.push_back(Event::Untrained { header: active.found.header });
            }
            if !active.framer.is_done() {
                self.end_burst(&active, true);
            }
        }
    }

    fn on_found(&mut self, found: Found) {
        if let Some(active) = &self.burst {
            // The same preamble again, from a neighbouring detector.
            if (found.start - active.found.start).abs() < 4.0 * found.profile.sps() {
                if !active.trained && found.quality > active.found.quality && found.profile == active.found.profile {
                    self.burst = None;
                    self.start_burst(found, false);
                } else {
                    self.ignored += 1;
                }
                return;
            }
            // A preamble inside the burst being read is the burst's own data
            // looking like one, which the header CRC makes rare; one near or
            // past its end, or while it is losing its pilots, is the next.
            let near_end = found.start > active.end - SLOT as f64 * found.profile.sps();
            let failing = active.trained && active.framer.pilots().last().is_some_and(|p| !p.found);
            if !near_end && !failing && active.trained {
                self.ignored += 1;
                return;
            }
            let mut active = self.burst.take().expect("checked above");
            if active.trained {
                active.framer.finish();
                self.framed(&mut active);
            }
            if !active.framer.is_done() || !active.trained {
                self.end_burst(&active, true);
            }
        }
        self.start_burst(found, true);
    }

    fn start_burst(&mut self, found: Found, announce: bool) {
        let header = found.header;
        let profile = found.profile;
        if announce {
            self.events.push_back(Event::Heard {
                profile,
                header,
                snr_db: found.snr_db,
                offset_hz: found.offset_hz(),
                seconds: found.start / FS,
            });
        }
        let voice = match header.kind {
            Kind::Voice => match VoiceMode::of(profile, &header) {
                Some(mode) => Some(mode),
                None => {
                    // A mode this receiver does not know: nothing to decode.
                    self.events.push_back(Event::Untrained { header });
                    return;
                }
            },
            Kind::Data => None,
        };
        let origin = (found.start.floor() as u64).saturating_sub(LEAD);
        if origin < self.raw_first {
            // Not kept any more: found too late to train on.
            self.events.push_back(Event::Untrained { header });
            return;
        }
        let options = Options { turn_limit_hz: TURN_LIMIT_HZ, least_trained_db: LEAST_TRAINED_DB, ..Options::fixed() };
        // Mixed down at the carrier the preamble measured, not the detector's:
        // what is left for the loop to follow is then only what moves after
        // it, which on a satellite pass is the Doppler still changing.
        let carrier = profile.carrier() + found.offset_hz();
        let mut core = Core::new(Band::new(FS, profile.baud(), carrier), options, Modulation::Bpsk.slicer());
        let from = (origin - self.raw_first) as usize;
        for &x in self.raw.range(from..) {
            core.feed(f64::from(x));
        }
        let Some(start) = core.half_near(found.start - origin as f64) else {
            self.events.push_back(Event::Untrained { header });
            return;
        };
        let geometry = header.geometry();
        core.train(Training {
            targets: found.symbols.clone(),
            start,
            first: Window { align: (SOLVE_FROM, PREAMBLE), solve: (SOLVE_FROM, PREAMBLE), search: SEARCH },
            retry: None,
            turn: Some(0.0),
            drift: None,
            accept_db: LEAST_TRAINED_DB,
            slicer: geometry.modulation.slicer(),
            fallback: false,
        });
        let noise = 10f64.powf(-found.snr_db / 10.0);
        let codewords = usize::from(header.codewords);
        let give_up = if voice.is_some() { GIVE_UP_VOICE } else { GIVE_UP_DATA };
        let end = found.start + burst_symbols(codewords) as f64 * profile.sps();
        self.burst = Some(Box::new(Active {
            framer: Framer::new(geometry.clone(), codewords, noise, give_up),
            geometry,
            voice,
            core,
            fed: self.taken,
            trained: false,
            ok: 0,
            failed: 0,
            slips: 0,
            end,
            found,
        }));
    }

    /// Give the burst's core what it has not had, and take its points.
    fn run_burst(&mut self) {
        let Some(mut active) = self.burst.take() else { return };
        while active.fed < self.taken {
            let x = self.raw[(active.fed - self.raw_first) as usize];
            active.core.feed(f64::from(x));
            active.fed += 1;
        }
        while let Some(heard) = active.core.heard() {
            match heard {
                Heard::Trained { snr_db, .. } => {
                    active.trained = true;
                    self.events.push_back(Event::Trained { snr_db });
                }
                Heard::Untrained => {
                    self.events.push_back(Event::Untrained { header: active.found.header });
                    return;
                }
                _ => {}
            }
        }
        if active.trained {
            while let Some(point) = active.core.next() {
                active.core.settle(point.nearest);
                active.framer.push(point.z);
                if active.framer.is_done() {
                    break;
                }
            }
            self.framed(&mut active);
            if active.framer.is_done() {
                return;
            }
        }
        // Nothing heard of it long after it should have ended.
        if (self.taken as f64) > active.end + FS {
            self.end_burst(&active, true);
            return;
        }
        self.burst = Some(active);
    }

    /// Pass on what the framer has to say.
    fn framed(&mut self, active: &mut Active) {
        let header = active.found.header;
        while let Some(event) = active.framer.next_event() {
            match event {
                Framed::Codeword { codeword, soft, erased_slots } => match active.voice {
                    Some(mode) => {
                        let decoded = soft.as_deref().and_then(|s| mode.decode(s));
                        if decoded.is_some() {
                            active.ok += 1;
                        } else {
                            active.failed += 1;
                        }
                        let end = decoded.as_ref().is_some_and(|c| c.end);
                        self.events.push_back(Event::Voice { mode, stream: header.stream, codeword: decoded, erased_slots });
                        if end {
                            // The speaker has let go: nothing more is coming.
                            active.framer.stop();
                        }
                    }
                    None => {
                        let block = soft.as_deref().and_then(|s| active.geometry.decode(s));
                        let index = block.as_ref().map(|b| b.0);
                        if block.is_some() {
                            active.ok += 1;
                        } else {
                            active.failed += 1;
                        }
                        self.events.push_back(Event::Block { header, codeword, block: index, erased_slots });
                        if let Some((index, data)) = block
                            && let Some(result) = self.assembler.add(&header, index, data)
                        {
                            self.events.push_back(Event::Delivered(result));
                        }
                    }
                },
                Framed::Slip { by, .. } => {
                    active.slips += 1;
                    self.events.push_back(Event::Slip { by });
                }
                Framed::Turned { .. } => {}
                Framed::Done { aborted } => self.end_burst(active, aborted),
            }
        }
    }

    fn end_burst(&mut self, active: &Active, aborted: bool) {
        self.last_points = active.framer.points().copied().collect();
        self.events.push_back(Event::BurstEnd(BurstReport {
            profile: active.found.profile,
            header: active.found.header,
            ok: active.ok,
            failed: active.failed,
            slips: active.slips,
            snr_db: active.framer.snr_db(),
            offset_hz: offset_of(active),
            drift_ppm: active.core.drift_ppm(),
            aborted,
        }));
    }
}

/// The far carrier's offset from the profile's, as the burst's core has it.
fn offset_of(active: &Active) -> f64 {
    active.core.band().carrier - active.found.profile.carrier() + active.core.offset_hz()
}
