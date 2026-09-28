//! From the equaliser's points back to codewords: finding each slot's
//! pilots, and trusting the data between two pilots only when both agree.
//!
//! BinModem's core follows a slip in fractions of a symbol and quarter
//! turns of phase, and says so; it does not know or care where the symbols
//! are in whole symbols, because nothing it was built for needed it to. A
//! VoIP jitter buffer that plays 20 ms of made-up audio moves everything
//! after it 48 symbols later -- a whole number of symbols and, at 1800 Hz,
//! a whole number of carrier cycles, so the core sees no jump at all -- and
//! one that drops 20 ms moves it 48 earlier. The pilots are what notice.
//!
//! Each slot's pilots are looked for within [`SEARCH`] symbols of where the
//! last slot's were found. Where they are says how far the stream has
//! moved; which way round they are says the constellation's phase, to a
//! multiple of the angle between its points. The data between two pilots
//! found in the same place and the same way round is demapped; the data
//! between two that disagree is erased -- its soft values set to nought, which
//! the Viterbi decoder reads as "unknown" -- because somewhere in it is the
//! slip, and every symbol after that is some other symbol. One slot of eight
//! erased is well within what the code fills in.

use std::collections::VecDeque;
use std::f64::consts::TAU;

use dsp::Complex;

use crate::frame::{Geometry, pilots};
use crate::profile::{DATA, PILOT, SLOT, SLOTS_PER_CODEWORD};

/// Symbols either side of where a slot's pilots are expected that they are
/// looked for: a 20 ms slip is 48, and the next slot's pilots are 128 away.
pub const SEARCH: i64 = 56;

/// Normalised pilot correlation, nought to one, for pilots to count as
/// found where they were expected, and for pilots somewhere else to be
/// believed instead. Sixteen pilots at 3 dB Es/N0 read about 0.8; random data
/// reads about 0.25, and above 0.7 about once in 2500.
const FOUND: f64 = 0.5;
const MOVED: f64 = 0.7;
/// How much better pilots somewhere else must read than where they were
/// expected.
const BETTER: f64 = 0.25;

/// Pilots missed in a row before a data burst is given up for lost: a
/// quarter of a second. A voice burst is given longer, to ride out a fade.
pub const GIVE_UP_DATA: usize = 3;
pub const GIVE_UP_VOICE: usize = 24;

/// What became of one slot's pilots.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Pilot {
    /// Whole symbols the stream had moved by here.
    pub shift: i64,
    /// The constellation's turn, in multiples of the angle between points.
    pub rotation: usize,
    pub found: bool,
    /// Normalised correlation where they were taken to be.
    pub quality: f64,
}

/// What the framer has to say.
#[derive(Debug, Clone, PartialEq)]
pub enum Framed {
    /// Codeword `codeword` of the burst: the soft values of its channel
    /// bits, None if every slot of it was erased or never came.
    Codeword { codeword: usize, soft: Option<Vec<f64>>, erased_slots: usize },
    /// The stream moved by `by` whole symbols between two slots.
    Slip { slot: usize, by: i64 },
    /// The constellation turned between two slots.
    Turned { slot: usize, rotation: usize },
    /// Every slot is done, or the burst was given up.
    Done { aborted: bool },
}

#[derive(Debug, Clone)]
pub struct Framer {
    geometry: Geometry,
    codewords: usize,
    slots: usize,
    /// Symbols from the equaliser, and the stream index of the first kept.
    stream: VecDeque<Complex>,
    first: i64,
    received: i64,
    /// Pilots resolved so far, in slot order.
    pilots: Vec<Pilot>,
    /// Data slots demapped so far, and the current codeword's soft values.
    demapped: usize,
    soft: Vec<f64>,
    erased: usize,
    /// Mean squared pilot error: the noise the soft values are scaled by.
    noise: f64,
    missed: usize,
    give_up: usize,
    done: bool,
    out: VecDeque<Framed>,
    /// Data points as they were decided, the constellation's turn taken out,
    /// for a display.
    points: VecDeque<Complex>,
}

/// Points kept for a display.
const POINTS: usize = 2048;

impl Framer {
    /// A framer for `codewords` codewords laid out by `geometry`, starting
    /// with an estimate of the noise, as a mean squared error, and giving up
    /// after `give_up` pilots missed in a row.
    pub fn new(geometry: Geometry, codewords: usize, noise: f64, give_up: usize) -> Self {
        Self {
            geometry,
            codewords,
            slots: codewords * SLOTS_PER_CODEWORD,
            stream: VecDeque::new(),
            first: 0,
            received: 0,
            pilots: Vec::new(),
            demapped: 0,
            soft: Vec::new(),
            erased: 0,
            noise: noise.clamp(1e-4, 1.0),
            missed: 0,
            give_up,
            done: false,
            out: VecDeque::new(),
            points: VecDeque::with_capacity(POINTS),
        }
    }

    pub fn is_done(&self) -> bool {
        self.done
    }

    /// Symbols the framer still wants before it has everything: the rest of
    /// the burst where it is expected, and the search's reach past it.
    pub fn wanted(&self) -> i64 {
        let shift = self.pilots.last().map_or(0, |p| p.shift);
        ((self.slots * SLOT + PILOT) as i64 + shift + SEARCH - self.received).max(0)
    }

    /// The next thing to report.
    pub fn next_event(&mut self) -> Option<Framed> {
        self.out.pop_front()
    }

    /// The noise as the pilots measure it, as Es/N0 in decibels.
    pub fn snr_db(&self) -> f64 {
        -10.0 * self.noise.log10()
    }

    /// The most recent data points, turned as the pilots say.
    pub fn points(&self) -> impl Iterator<Item = &Complex> {
        self.points.iter()
    }

    pub fn pilots(&self) -> &[Pilot] {
        &self.pilots
    }

    /// Take the next point from the equaliser.
    pub fn push(&mut self, z: Complex) {
        if self.done {
            return;
        }
        self.stream.push_back(z);
        self.received += 1;
        while !self.done && self.resolve_next() {}
    }

    /// The burst is over before its announced end -- a voice transmission
    /// whose speaker let go -- and nothing is missing.
    pub fn stop(&mut self) {
        if !self.done {
            self.done = true;
            self.out.push_back(Framed::Done { aborted: false });
        }
    }

    /// No more points are coming: finish with what there is, the rest
    /// unknown.
    pub fn finish(&mut self) {
        while !self.done {
            self.stream.push_back(Complex::ZERO);
            self.received += 1;
            while !self.done && self.resolve_next() {}
        }
    }

    fn point(&self, index: i64) -> Complex {
        index
            .checked_sub(self.first)
            .and_then(|i| usize::try_from(i).ok())
            .and_then(|i| self.stream.get(i).copied())
            .unwrap_or(Complex::ZERO)
    }

    /// Resolve the next pilot if its symbols are all in, and demap the slot
    /// before it. False if there is nothing more to do yet.
    fn resolve_next(&mut self) -> bool {
        let slot = self.pilots.len();
        if slot > self.slots {
            return false;
        }
        let expected = self.pilots.last().map_or(0, |p| p.shift);
        let nominal = (slot * SLOT) as i64;
        if self.received < nominal + expected + SEARCH + PILOT as i64 {
            return false;
        }
        let pilot = self.find(slot, expected);
        if pilot.found {
            self.missed = 0;
        } else {
            self.missed += 1;
        }
        if let Some(before) = self.pilots.last() {
            if pilot.shift != before.shift {
                self.out.push_back(Framed::Slip { slot, by: pilot.shift - before.shift });
            } else if pilot.found && before.found && pilot.rotation != before.rotation {
                self.out.push_back(Framed::Turned { slot, rotation: pilot.rotation });
            }
        }
        self.pilots.push(pilot);
        if slot > 0 {
            self.demap(slot - 1);
        }
        if slot == self.slots {
            self.finish_burst(false);
        } else if self.missed >= self.give_up {
            self.finish_burst(true);
        }
        // Keep only what the next search can reach.
        let keep_from = ((slot + 1) * SLOT) as i64 + pilot.shift - SEARCH - SLOT as i64;
        while self.first < keep_from && !self.stream.is_empty() {
            self.stream.pop_front();
            self.first += 1;
        }
        true
    }

    /// Look for slot `slot`'s pilots around shift `expected`.
    fn find(&self, slot: usize, expected: i64) -> Pilot {
        let chips = pilots(slot);
        let nominal = (slot * SLOT) as i64;
        let read = |shift: i64| {
            let (mut c, mut power) = (Complex::ZERO, 0.0);
            for (j, &chip) in chips.iter().enumerate() {
                let z = self.point(nominal + shift + j as i64);
                c += z.scale(chip);
                power += z.norm_sqr();
            }
            let quality = c.abs() / (PILOT as f64 * power).sqrt().max(1e-30);
            (c, quality)
        };
        let (here, here_quality) = read(expected);
        let (best_shift, (best, best_quality)) = (expected - SEARCH..=expected + SEARCH)
            .map(|s| (s, read(s)))
            .max_by(|a, b| a.1.1.total_cmp(&b.1.1))
            .unwrap_or((expected, (here, here_quality)));
        let (shift, c, quality, found) = if best_shift != expected
            && best_quality >= MOVED
            && best_quality > here_quality + BETTER
        {
            (best_shift, best, best_quality, true)
        } else {
            (expected, here, here_quality, here_quality >= FOUND)
        };
        let points = self.geometry.modulation.points();
        let rotation = if found {
            ((c.arg() / TAU * points as f64).round() as i64).rem_euclid(points as i64) as usize
        } else {
            self.pilots.last().map_or(0, |p| p.rotation)
        };
        Pilot { shift, rotation, found, quality }
    }

    /// Demap data slot `slot`, between pilots `slot` and `slot + 1`, or
    /// erase it if they disagree.
    fn demap(&mut self, slot: usize) {
        let (before, after) = (self.pilots[slot], self.pilots[slot + 1]);
        let modulation = self.geometry.modulation;
        let agree = before.shift == after.shift && !(before.found && after.found && before.rotation != after.rotation);
        if agree {
            let rotation = if before.found { before.rotation } else { after.rotation };
            let spin = Complex::from_polar(1.0, -TAU * rotation as f64 / modulation.points() as f64);
            let start = (slot * SLOT + PILOT) as i64 + before.shift;
            // The pilots' own error, which is the noise the soft values need.
            if before.found {
                let chips = pilots(slot);
                let error: f64 = chips
                    .iter()
                    .enumerate()
                    .map(|(j, &p)| (self.point((slot * SLOT) as i64 + before.shift + j as i64) * spin - Complex::new(p, 0.0)).norm_sqr())
                    .sum::<f64>()
                    / PILOT as f64;
                self.noise += 0.2 * (error.clamp(1e-4, 2.0) - self.noise);
            }
            for i in 0..DATA as i64 {
                let z = self.point(start + i) * spin;
                modulation.demap(z, self.noise, &mut self.soft);
                if self.points.len() >= POINTS {
                    self.points.pop_front();
                }
                self.points.push_back(z);
            }
        } else {
            self.soft.extend(std::iter::repeat_n(0.0, DATA * modulation.bits()));
            self.erased += 1;
        }
        self.demapped = slot + 1;
        if self.demapped.is_multiple_of(SLOTS_PER_CODEWORD) {
            self.decode(self.demapped / SLOTS_PER_CODEWORD - 1);
        }
    }

    fn decode(&mut self, codeword: usize) {
        let soft = std::mem::take(&mut self.soft);
        let soft = (self.erased < SLOTS_PER_CODEWORD).then_some(soft);
        self.out.push_back(Framed::Codeword { codeword, soft, erased_slots: self.erased });
        self.erased = 0;
    }

    fn finish_burst(&mut self, aborted: bool) {
        if aborted {
            // The codeword under way, with what it has; the rest are lost.
            let done = self.demapped / SLOTS_PER_CODEWORD;
            if !self.demapped.is_multiple_of(SLOTS_PER_CODEWORD) {
                let wanted = self.geometry.channel_bits;
                let have = self.soft.len();
                self.soft.extend(std::iter::repeat_n(0.0, wanted.saturating_sub(have)));
                self.erased += SLOTS_PER_CODEWORD - self.demapped % SLOTS_PER_CODEWORD;
                self.decode(done);
            }
            let next = self.demapped.div_ceil(SLOTS_PER_CODEWORD);
            for codeword in next..self.codewords {
                self.out.push_back(Framed::Codeword { codeword, soft: None, erased_slots: SLOTS_PER_CODEWORD });
            }
        }
        self.done = true;
        self.out.push_back(Framed::Done { aborted });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fec::Rate;
    use crate::frame::{Header, Kind, burst};
    use crate::profile::PREAMBLE;
    use crate::psk::Modulation;

    fn sent(modulation: Modulation, codewords: usize) -> (Geometry, Vec<Complex>, Vec<Vec<u8>>) {
        let geometry = Geometry::new(modulation, Rate::Half);
        let header = Header {
            kind: Kind::Data,
            modulation,
            rate: Rate::Half,
            codec: 0,
            codewords: codewords as u8,
            stream: 1,
            sequence: 0,
            total: codewords as u16,
        };
        let payloads: Vec<Vec<u8>> =
            (0..codewords).map(|c| (0..geometry.payload).map(|i| (i * 13 + c * 7) as u8).collect()).collect();
        let cws: Vec<Vec<Complex>> = payloads.iter().enumerate().map(|(i, p)| geometry.encode(i as u16, p)).collect();
        let symbols = burst(header, &cws)[PREAMBLE..].to_vec();
        (geometry, symbols, payloads)
    }

    fn run(framer: &mut Framer, symbols: &[Complex]) -> Vec<Framed> {
        for &z in symbols {
            framer.push(z);
        }
        framer.finish();
        std::iter::from_fn(|| framer.next_event()).collect()
    }

    fn blocks(g: &Geometry, events: &[Framed]) -> Vec<Option<(u16, Vec<u8>)>> {
        events
            .iter()
            .filter_map(|e| match e {
                Framed::Codeword { soft, .. } => Some(soft.as_ref().and_then(|s| g.decode(s))),
                _ => None,
            })
            .collect()
    }

    fn decoded(g: &Geometry, events: &[Framed]) -> Vec<Option<u16>> {
        blocks(g, events).into_iter().map(|b| b.map(|b| b.0)).collect()
    }

    #[test]
    fn clean_symbols_decode() {
        for m in Modulation::ALL {
            let (g, symbols, payloads) = sent(m, 3);
            let mut framer = Framer::new(g.clone(), 3, 0.01, GIVE_UP_DATA);
            let events = run(&mut framer, &symbols);
            assert_eq!(decoded(&g, &events), vec![Some(0), Some(1), Some(2)], "{}", m.label());
            assert_eq!(blocks(&g, &events)[0].as_ref().map(|b| &b.1), Some(&payloads[0]));
        }
    }

    #[test]
    fn a_turned_constellation_is_turned_back() {
        let (g, symbols, _) = sent(Modulation::Psk8, 2);
        let turn = Complex::from_polar(1.0, 3.0 * TAU / 8.0);
        let turned: Vec<Complex> = symbols.iter().map(|z| *z * turn).collect();
        let mut framer = Framer::new(g.clone(), 2, 0.01, GIVE_UP_DATA);
        assert_eq!(decoded(&g, &run(&mut framer, &turned)), vec![Some(0), Some(1)]);
    }

    #[test]
    fn a_slip_either_way_costs_one_slot() {
        for m in Modulation::ALL {
            for by in [48i64, -48] {
                let (g, symbols, _) = sent(m, 3);
                let at = 3 * SLOT + 40;
                let mut slipped = symbols[..at].to_vec();
                if by > 0 {
                    // 20 ms of something else inserted.
                    slipped.extend((0..by).map(|i| Complex::from_polar(1.0, i as f64)));
                    slipped.extend_from_slice(&symbols[at..]);
                } else {
                    slipped.extend_from_slice(&symbols[at + (-by) as usize..]);
                }
                let mut framer = Framer::new(g.clone(), 3, 0.01, GIVE_UP_DATA);
                let events = run(&mut framer, &slipped);
                assert!(events.contains(&Framed::Slip { slot: 4, by }), "{} {by}: {events:?}", m.label());
                assert_eq!(decoded(&g, &events), vec![Some(0), Some(1), Some(2)], "{} {by}", m.label());
            }
        }
    }

    #[test]
    fn a_burst_that_stops_is_given_up() {
        let (g, symbols, _) = sent(Modulation::Qpsk, 4);
        let mut framer = Framer::new(g.clone(), 4, 0.01, GIVE_UP_DATA);
        let events = run(&mut framer, &symbols[..SLOT * 9]);
        assert_eq!(decoded(&g, &events), vec![Some(0), None, None, None]);
        assert_eq!(events.last(), Some(&Framed::Done { aborted: true }));
    }
}
