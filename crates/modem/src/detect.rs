//! Finding a preamble, and reading its header, before anything is trained.
//!
//! The line is mixed down with a fixed oscillator and matched-filtered four
//! times a symbol. 16 kHz is 6 2/3 samples a symbol, so four outputs a
//! symbol fall every 1 2/3 samples: three fractional positions in turn, each
//! with its own set of filter taps. Nothing is resampled, so the time of
//! every output is exact and the detector's timing is the line's.
//!
//! Each output is multiplied by the conjugate of the one a symbol before it.
//! On the preamble that product is the differential bit -- +1 for a repeat,
//! -1 for a reversal -- turned by however far the carrier moves in a symbol,
//! the same for every product. So correlating the products against the
//! unique word's signs gives a peak whose size does not depend on the
//! carrier's phase or offset, and whose angle is that offset. The
//! correlation is normalised by the products' own magnitudes, so the
//! threshold means the same at every level.
//!
//! Of the four outputs a symbol one lies nearest the symbol centres. The
//! peak's shape across its neighbours, about a quarter of a symbol apart,
//! places the centres between them; the header is then read, differentially,
//! from outputs interpolated at those centres. A header whose CRC holds
//! makes the whole preamble known, and the carrier's offset is measured
//! again across all of it, far more finely than the unique word alone can.

use std::collections::VecDeque;
use std::f64::consts::TAU;

use dsp::Complex;

use crate::frame::{Header, preamble, unique_word_signs};
use crate::profile::{FS, HEADER_SYMBOLS, PREAMBLE, Profile, UW, UW_LAST};
use crate::tx::shaped_pulse;

/// Matched-filter outputs a symbol.
const OVER: usize = 4;

/// Symbols either side the matched filter reaches.
const SPAN: f64 = 8.0;

/// The normalised correlation that starts a look for a peak. Noise alone
/// reaches it about once a second at 9600 outputs a second, and the
/// header's CRC turns those away; a preamble at 3 dB Es/N0 is well above it.
const TRIGGER: f64 = 0.5;

/// The least mean product magnitude worth looking at: about -70 dBFS.
const QUIET: f64 = 1e-7;

/// Outputs after a trigger over which the peak is looked for: a symbol and
/// a half.
const LOOK: u64 = 6;

/// Matched-filter outputs kept: enough to reach back from the end of the
/// header to the start of the preamble, and some.
const KEPT: usize = 2048;

/// A preamble found, with its header read.
#[derive(Debug, Clone, PartialEq)]
pub struct Found {
    pub profile: Profile,
    /// Where the centre of the preamble's first symbol fell, in samples since
    /// the detector's first.
    pub start: f64,
    /// The carrier the detector mixed down with, and how far the far carrier
    /// turns in a symbol against it, in radians.
    pub mixer_hz: f64,
    pub turn: f64,
    pub header: Header,
    /// Every symbol of the preamble, known now the header is.
    pub symbols: Vec<Complex>,
    /// The unique word's normalised correlation, nought to one.
    pub quality: f64,
    /// Es/N0 of the preamble as the matched filter saw it, in decibels,
    /// against the known symbols.
    pub snr_db: f64,
}

impl Found {
    /// The far carrier's offset from the profile's, in hertz.
    pub fn offset_hz(&self) -> f64 {
        self.mixer_hz - self.profile.carrier() + self.turn * self.profile.baud() / TAU
    }
}

/// A peak being looked for.
#[derive(Debug, Clone, Copy)]
struct Peak {
    best: u64,
    size: f64,
    until: u64,
}

/// A peak found, waiting for its header to arrive.
#[derive(Debug, Clone, Copy)]
struct Candidate {
    /// Output index, fractional, of the unique word's last symbol.
    at: f64,
    turn: f64,
    quality: f64,
    ready: u64,
}

#[derive(Debug, Clone)]
pub struct Detector {
    profile: Profile,
    /// The carrier mixed down with, and the mixer's phase, in turns, and its
    /// step a sample.
    mixer_hz: f64,
    phase: f64,
    step: f64,
    /// Mixed-down samples, and the index of the oldest; samples taken.
    base: VecDeque<Complex>,
    base_first: i64,
    taken: u64,
    /// Outputs fall every `num / den` samples: 5/2 narrow, 5/3 wide.
    num: u64,
    den: u64,
    /// Matched-filter taps for each of the `den` fractional positions, and
    /// how far they reach before the sample an output falls after.
    taps: Vec<Vec<f64>>,
    reach: i64,
    /// Outputs made, and the kept ones with the index of the oldest.
    made: u64,
    mf: VecDeque<Complex>,
    mf_first: u64,
    /// Each output times the conjugate of the one a symbol before, and the
    /// unique word's correlation ending at each output.
    products: VecDeque<Complex>,
    correlation: VecDeque<Complex>,
    signs: Vec<f64>,
    peak: Option<Peak>,
    candidates: Vec<Candidate>,
    /// No new peak is looked for before this output.
    holdoff: u64,
    found: VecDeque<Found>,
    /// Peaks whose header failed its CRC: noise, or a preamble too damaged.
    rejected: u64,
}

fn gcd(mut a: u64, mut b: u64) -> u64 {
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a
}

impl Detector {
    /// A detector for `profile`'s preambles, mixing down with its carrier
    /// moved by `offset_hz`.
    pub fn new(profile: Profile, offset_hz: f64) -> Self {
        let sps = profile.sps();
        // The output spacing, a quarter symbol, as a fraction in samples.
        let (fs, per) = (FS as u64, OVER as u64 * profile.baud() as u64);
        let g = gcd(fs, per);
        let (num, den) = (fs / g, per / g);
        let reach = (SPAN * sps).ceil() as i64;
        let taps = (0..den)
            .map(|p| {
                let frac = p as f64 / den as f64;
                (-reach..=reach + 1)
                    .map(|m| {
                        let t = (frac - m as f64) / sps;
                        // Scaled so that a symbol of amplitude A on the line
                        // comes out at A at its centre.
                        shaped_pulse(t, SPAN) / sps
                    })
                    .collect()
            })
            .collect();
        let mixer_hz = profile.carrier() + offset_hz;
        Self {
            profile,
            mixer_hz,
            phase: 0.0,
            step: mixer_hz / FS,
            base: VecDeque::from(vec![Complex::ZERO; reach as usize + 1]),
            base_first: -(reach + 1),
            taken: 0,
            num,
            den,
            taps,
            reach,
            made: 0,
            mf: VecDeque::with_capacity(KEPT),
            mf_first: 0,
            products: VecDeque::with_capacity(KEPT),
            correlation: VecDeque::with_capacity(KEPT),
            signs: unique_word_signs(),
            peak: None,
            candidates: Vec::new(),
            holdoff: 0,
            found: VecDeque::new(),
            rejected: 0,
        }
    }

    pub fn profile(&self) -> Profile {
        self.profile
    }

    /// Samples taken so far.
    pub fn taken(&self) -> u64 {
        self.taken
    }

    /// Peaks turned away because their header did not check.
    pub fn rejected(&self) -> u64 {
        self.rejected
    }

    /// The next preamble found, if there is one.
    pub fn found(&mut self) -> Option<Found> {
        self.found.pop_front()
    }

    /// Take one line sample.
    pub fn feed(&mut self, x: f64) {
        let angle = TAU * self.phase;
        self.base.push_back(Complex::new(angle.cos(), -angle.sin()).scale(2.0 * x));
        self.phase += self.step;
        self.phase -= self.phase.floor();
        self.taken += 1;
        loop {
            // Output k falls at k num / den samples.
            let at = self.num * self.made;
            let (whole, frac) = ((at / self.den) as i64, (at % self.den) as usize);
            if whole + self.reach + 1 >= self.taken as i64 {
                break;
            }
            let from = whole - self.reach - self.base_first;
            debug_assert!(from >= 0, "the filter reaches before the samples kept");
            let y = self.taps[frac]
                .iter()
                .enumerate()
                .fold(Complex::ZERO, |sum, (i, &h)| sum + self.base[from as usize + i].scale(h));
            self.output(y);
            // Keep only what the next output needs.
            let next = (self.num * self.made / self.den) as i64 - self.reach;
            while self.base_first < next && !self.base.is_empty() {
                self.base.pop_front();
                self.base_first += 1;
            }
        }
    }

    fn output(&mut self, y: Complex) {
        let k = self.made;
        self.made += 1;
        let before = self.mf_at(k.wrapping_sub(OVER as u64));
        self.mf.push_back(y);
        let product = before.map_or(Complex::ZERO, |b| y * b.conj());
        self.products.push_back(product);

        // The unique word's correlation ending here.
        let reach = (OVER * (UW - 1)) as u64;
        let (mut c, mut e) = (Complex::ZERO, 0.0);
        if k >= reach + OVER as u64 {
            let last = self.products.len() - 1;
            for (j, &sign) in self.signs.iter().enumerate() {
                let d = self.products[last - OVER * (UW - 1 - j)];
                c += d.scale(sign);
                e += d.abs();
            }
        }
        self.correlation.push_back(c);
        while self.mf.len() > KEPT {
            self.mf.pop_front();
            self.products.pop_front();
            self.correlation.pop_front();
            self.mf_first += 1;
        }

        let quality = c.abs() / (e + 1e-30);
        if quality > TRIGGER && e / UW as f64 > QUIET && k >= self.holdoff {
            match &mut self.peak {
                Some(peak) => {
                    if c.abs() > peak.size {
                        peak.best = k;
                        peak.size = c.abs();
                    }
                }
                None => self.peak = Some(Peak { best: k, size: c.abs(), until: k + LOOK }),
            }
        }
        if let Some(peak) = self.peak
            && k >= peak.until.max(peak.best + 1)
        {
            self.peak = None;
            self.place(peak);
        }

        let ready: Vec<Candidate> = self.candidates.iter().copied().filter(|c| k >= c.ready).collect();
        self.candidates.retain(|c| k < c.ready);
        for candidate in ready {
            self.read(candidate);
        }
    }

    /// Output `k`, if it is still kept.
    fn mf_at(&self, k: u64) -> Option<Complex> {
        let i = k.checked_sub(self.mf_first)?;
        self.mf.get(i as usize).copied()
    }

    fn correlation_at(&self, k: u64) -> Option<Complex> {
        let i = k.checked_sub(self.mf_first)?;
        self.correlation.get(i as usize).copied()
    }

    /// The matched filter's output at fractional output index `q`, by cubic
    /// interpolation between the outputs either side.
    fn interpolate(&self, q: f64) -> Option<Complex> {
        let i = q.floor() as i64;
        let f = q - q.floor();
        let at = |d: i64| u64::try_from(i + d).ok().and_then(|k| self.mf_at(k));
        let (p0, p1, p2, p3) = (at(-1)?, at(0)?, at(1)?, at(2)?);
        // Catmull-Rom.
        let a = p1.scale(2.0);
        let b = (p2 - p0).scale(f);
        let c = (p0.scale(2.0) - p1.scale(5.0) + p2.scale(4.0) - p3).scale(f * f);
        let d = (p1.scale(3.0) - p0 - p2.scale(3.0) + p3).scale(f * f * f);
        Some((a + b + c + d).scale(0.5))
    }

    /// Place a peak between its neighbours and wait for its header.
    fn place(&mut self, peak: Peak) {
        let size = |k: u64| self.correlation_at(k).map_or(0.0, |c| c.abs());
        let (early, centre, late) = (size(peak.best.saturating_sub(1)), size(peak.best), size(peak.best + 1));
        let curve = early - 2.0 * centre + late;
        let offset = if curve < 0.0 { (0.5 * (early - late) / curve).clamp(-0.5, 0.5) } else { 0.0 };
        let c = self.correlation_at(peak.best).unwrap_or(Complex::ZERO);
        let at = peak.best as f64 + offset;
        let e: f64 = {
            let last = peak.best;
            (0..UW)
                .filter_map(|j| {
                    let k = last.checked_sub((OVER * (UW - 1 - j)) as u64)?;
                    let i = k.checked_sub(self.mf_first)?;
                    self.products.get(i as usize).map(|d| d.abs())
                })
                .sum()
        };
        self.candidates.push(Candidate {
            at,
            turn: c.arg(),
            quality: c.abs() / (e + 1e-30),
            // The header's last symbol and the two outputs past it that the
            // interpolation reads.
            ready: peak.best + (OVER * HEADER_SYMBOLS) as u64 + 4,
        });
        self.holdoff = peak.best + 2 * OVER as u64;
    }

    /// Read a candidate's header, and if it checks, everything else.
    fn read(&mut self, candidate: Candidate) {
        let spin = Complex::from_polar(1.0, -candidate.turn);
        let products: Option<Vec<Complex>> = (1..=HEADER_SYMBOLS)
            .map(|i| {
                let q = candidate.at + (OVER * i) as f64;
                Some(self.interpolate(q)? * self.interpolate(q - OVER as f64)?.conj() * spin)
            })
            .collect();
        let Some(products) = products else {
            self.rejected += 1;
            return;
        };
        let scale = products.iter().map(|d| d.abs()).sum::<f64>() / products.len() as f64;
        let soft: Vec<f64> = products.iter().map(|d| d.re / (scale + 1e-30)).collect();
        let Some(header) = Header::decode(&soft) else {
            self.rejected += 1;
            return;
        };

        // Every preamble symbol is known now: measure the turn across all of
        // them, roughly a symbol apart and then finely sixteen apart.
        let symbols = preamble(header);
        let first = candidate.at - (OVER * UW_LAST) as f64;
        let stripped: Option<Vec<Complex>> = symbols
            .iter()
            .enumerate()
            .map(|(i, a)| Some(self.interpolate(first + (OVER * i) as f64)? * a.conj()))
            .collect();
        let Some(stripped) = stripped else {
            self.rejected += 1;
            return;
        };
        let lag = |l: usize| (l..PREAMBLE).fold(Complex::ZERO, |sum, i| sum + stripped[i] * stripped[i - l].conj());
        let rough = lag(1).arg();
        let fine = lag(16).arg();
        let wraps = ((16.0 * rough - fine) / TAU).round();
        let turn = (fine + TAU * wraps) / 16.0;

        // Signal to noise against the known symbols, the turn taken out and
        // the phase and gain fitted.
        let turned: Vec<Complex> =
            stripped.iter().enumerate().map(|(i, w)| *w * Complex::from_polar(1.0, -turn * i as f64)).collect();
        let mean = turned.iter().fold(Complex::ZERO, |s, w| s + *w).scale(1.0 / PREAMBLE as f64);
        let noise = turned.iter().map(|w| (*w - mean).norm_sqr()).sum::<f64>() / PREAMBLE as f64;
        let snr_db = 10.0 * (mean.norm_sqr() / noise.max(1e-30)).log10();

        self.found.push_back(Found {
            profile: self.profile,
            start: first * self.num as f64 / self.den as f64,
            mixer_hz: self.mixer_hz,
            turn,
            header,
            symbols,
            quality: candidate.quality,
            snr_db,
        });
        // Nothing in the header is a unique word.
        self.holdoff = self.holdoff.max(candidate.at as u64 + (OVER * HEADER_SYMBOLS) as u64);
        // And a peak inside the preamble just read is the same preamble.
        self.candidates.retain(|c| (c.at - candidate.at).abs() > (OVER * HEADER_SYMBOLS) as f64);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::channel::Random;
    use crate::fec::Rate;
    use crate::frame::Kind;
    use crate::psk::Modulation;
    use crate::tx::Modulator;

    fn header() -> Header {
        Header {
            kind: Kind::Data,
            modulation: Modulation::Qpsk,
            rate: Rate::Half,
            codec: 0,
            codewords: 3,
            stream: 42,
            sequence: 0,
            total: 3,
        }
    }

    fn noise(samples: &mut [f32], sigma: f64, seed: u64) {
        let mut random = Random::new(seed);
        for s in samples {
            *s += (sigma * random.gaussian()) as f32;
        }
    }

    /// The preamble of `header` after `lead` samples of silence, sent in
    /// `profile` with the carrier `offset_hz` out.
    fn line(profile: Profile, lead: usize, offset_hz: f64) -> Vec<f32> {
        let mut m = Modulator::new(profile, -15.0);
        m.misalign(offset_hz, 0.0);
        let mut out = Vec::new();
        m.fill(lead, &mut out);
        m.push(&preamble(header()));
        out.extend(m.drain());
        out.extend(std::iter::repeat_n(0.0, 2000));
        out
    }

    /// Every preamble found by `profile`'s detector bank.
    fn detect(profile: Profile, samples: &[f32]) -> Vec<Found> {
        let mut bank: Vec<Detector> = profile.search_offsets().iter().map(|&o| Detector::new(profile, o)).collect();
        let mut found = Vec::new();
        for &x in samples {
            for d in &mut bank {
                d.feed(f64::from(x));
                while let Some(f) = d.found() {
                    found.push(f);
                }
            }
        }
        found
    }

    /// Where the modulator put the first symbol's centre after `lead`
    /// samples: its push starts the pulse, ten symbols before the centre.
    fn true_start(profile: Profile, lead: usize) -> f64 {
        ((lead as f64 / profile.sps()).ceil() + 10.0) * profile.sps()
    }

    #[test]
    fn a_clean_preamble_is_found_where_it_is() {
        for profile in Profile::ALL {
            for lead in [1000, 1003, 1007] {
                let found = detect(profile, &line(profile, lead, 0.0));
                let f = found.iter().max_by(|a, b| a.quality.total_cmp(&b.quality)).expect("found");
                assert_eq!(f.profile, profile);
                assert_eq!(f.header, header());
                let error = f.start - true_start(profile, lead);
                assert!(error.abs() < 0.5, "{} lead {lead}: start {error:.2} samples out", profile.name());
                assert!(f.offset_hz().abs() < 0.5, "offset {:.2} Hz", f.offset_hz());
                assert!(f.snr_db > 30.0, "snr {:.1}", f.snr_db);
            }
        }
    }

    #[test]
    fn the_carrier_offset_is_measured() {
        for (profile, offsets) in [(Profile::Wide, &[-60.0, -7.0, 13.0, 90.0][..]), (Profile::Narrow, &[-420.0, -150.0, 35.0, 260.0][..])] {
            for &offset in offsets {
                let found = detect(profile, &line(profile, 800, offset));
                assert!(!found.is_empty(), "{} offset {offset}", profile.name());
                let f = found.iter().max_by(|a, b| a.quality.total_cmp(&b.quality)).unwrap();
                assert!((f.offset_hz() - offset).abs() < 0.5, "{}: {:.2} Hz for {offset}", profile.name(), f.offset_hz());
            }
        }
    }

    #[test]
    fn a_noisy_preamble_is_still_found() {
        // 4 dB Es/N0: noise across the 8 kHz band at the level that puts that
        // much in a symbol's bandwidth.
        for profile in Profile::ALL {
            let mut samples = line(profile, 1500, 5.0);
            let signal = 10f64.powf(-15.0 / 20.0) / 2f64.sqrt();
            let sigma = signal * (FS / 2.0 / profile.baud()).sqrt() * 10f64.powf(-4.0 / 20.0);
            noise(&mut samples, sigma, 9);
            let found = detect(profile, &samples);
            let f = found.iter().max_by(|a, b| a.quality.total_cmp(&b.quality)).expect("found");
            assert!((f.start - true_start(profile, 1500)).abs() < 1.5, "{}", profile.name());
            assert!((f.offset_hz() - 5.0).abs() < 3.0, "{}", profile.name());
        }
    }

    #[test]
    fn neither_profile_mistakes_the_other() {
        assert!(detect(Profile::Narrow, &line(Profile::Wide, 900, 0.0)).is_empty());
        assert!(detect(Profile::Wide, &line(Profile::Narrow, 900, 0.0)).is_empty());
    }

    #[test]
    fn noise_alone_finds_nothing() {
        let mut samples = vec![0.0f32; 16_000 * 5];
        noise(&mut samples, 0.05, 3);
        for profile in Profile::ALL {
            assert!(detect(profile, &samples).is_empty());
        }
    }
}
