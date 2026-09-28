//! The transmitter: symbols to line samples.
//!
//! Root-raised-cosine pulses on the carrier, evaluated at each sample's own
//! position in symbol time rather than from a fixed tap table, because 16 kHz
//! is 6 2/3 samples a symbol and no table indexed by whole samples fits it.
//!
//! It is a source that is drained, not a function from a burst to a buffer: a
//! transfer can be many minutes long, and the line takes it a few hundred
//! samples at a time. Symbols are queued a burst at a time; between bursts it
//! sends silence and says it is idle.

use std::collections::VecDeque;
use std::f64::consts::{PI, TAU};
use std::sync::OnceLock;

use dsp::{Complex, rrc_at};

use crate::profile::{FS, Profile, ROLLOFF};

/// Symbols either side of its centre a pulse reaches, and its table's steps
/// a symbol.
const SPAN: i64 = 10;
const STEPS: usize = 256;

/// Symbols at each end of a pulse over which it is tapered to nothing.
const TAPER: f64 = 3.0;

/// The root-raised-cosine pulse at `t` symbols from its centre, cut off at
/// `span` symbols either side and tapered over the last [`TAPER`] of them.
///
/// Flat in the middle and tapered only at the ends: a Hann window across the
/// whole span, as BinModem's test modulator has, bends the pulse near its
/// centre where the zero crossings are, and at a fifth of roll-off that left
/// a clean line reading 23 dB at the matched filter. Tapered like this it
/// reads above 40.
pub fn shaped_pulse(t: f64, span: f64) -> f64 {
    let a = t.abs();
    if a >= span {
        return 0.0;
    }
    let flat = span - TAPER;
    let window = if a <= flat { 1.0 } else { 0.5 + 0.5 * (PI * (a - flat) / TAPER).cos() };
    rrc_at(t, ROLLOFF) * window
}

/// The pulse, tabulated.
fn pulse() -> &'static [f64] {
    static PULSE: OnceLock<Vec<f64>> = OnceLock::new();
    PULSE.get_or_init(|| {
        (0..=2 * SPAN as usize * STEPS + 1)
            .map(|i| shaped_pulse(i as f64 / STEPS as f64 - SPAN as f64, SPAN as f64))
            .collect()
    })
}

fn pulse_at(t: f64) -> f64 {
    let x = (t + SPAN as f64) * STEPS as f64;
    if x < 0.0 {
        return 0.0;
    }
    let (i, frac) = (x.floor() as usize, x - x.floor());
    let table = pulse();
    if i + 1 >= table.len() {
        return 0.0;
    }
    table[i] * (1.0 - frac) + table[i + 1] * frac
}

/// A modulator with a queue of symbols in front of it.
#[derive(Debug, Clone)]
pub struct Modulator {
    profile: Profile,
    /// Peak of a lone pulse on the carrier: the line's amplitude for symbols
    /// at unit power.
    amplitude: f64,
    /// Symbols waiting and going out, and the index of the first of them in
    /// the modulator's own count of symbol periods.
    queue: VecDeque<Complex>,
    first: i64,
    /// Samples made.
    sample: u64,
    /// The carrier, and the symbol clock against nominal: the profile's and
    /// one, except to simulate a far end that is not quite right.
    carrier: f64,
    clock: f64,
}

impl Modulator {
    /// A modulator for `profile` whose signal is `level_dbfs` rms, relative
    /// to a full-scale sine. At -12 dBFS the peaks of shaped 8PSK sit near
    /// -5 dBFS.
    pub fn new(profile: Profile, level_dbfs: f64) -> Self {
        let mut m = Self {
            profile,
            amplitude: 0.0,
            queue: VecDeque::new(),
            first: 0,
            sample: 0,
            carrier: profile.carrier(),
            clock: 1.0,
        };
        m.set_level(level_dbfs);
        m
    }

    pub fn profile(&self) -> Profile {
        self.profile
    }

    /// Change profile. Anything queued is dropped: its symbols were for the
    /// other one.
    pub fn set_profile(&mut self, profile: Profile) {
        if profile != self.profile {
            self.queue.clear();
            self.profile = profile;
            self.carrier = profile.carrier();
            self.clock = 1.0;
        }
    }

    /// Set the rms level, in dB relative to a full-scale sine's rms.
    pub fn set_level(&mut self, level_dbfs: f64) {
        // Unit-power symbols on a unit-energy pulse are a complex envelope of
        // unit mean power, which on the carrier is a mean power of A^2 / 2:
        // the same as a sine of peak A.
        self.amplitude = 10f64.powf(level_dbfs / 20.0);
    }

    /// Misplace the carrier by `offset_hz` and run the symbol clock `ppm`
    /// fast, as a far end whose clocks differ from ours would. For tests.
    pub fn misalign(&mut self, offset_hz: f64, ppm: f64) {
        self.carrier = self.profile.carrier() + offset_hz;
        self.clock = 1.0 + ppm * 1e-6;
    }

    /// Where sample `n` falls in the modulator's symbol time.
    fn symbol_time(&self, n: u64) -> f64 {
        n as f64 / self.profile.sps() * self.clock
    }

    /// Queue a burst's symbols. If nothing is going out, the first pulse
    /// starts now rather than having been due in the past.
    pub fn push(&mut self, symbols: &[Complex]) {
        if self.queue.is_empty() {
            self.first = self.symbol_time(self.sample).ceil() as i64 + SPAN;
        }
        self.queue.extend(symbols.iter().copied());
    }

    /// Whether anything is still to go out, pulse tails included.
    pub fn busy(&self) -> bool {
        !self.queue.is_empty()
    }

    /// Symbols still queued, including those partly sent.
    pub fn queued(&self) -> usize {
        self.queue.len()
    }

    /// Abandon whatever is queued. The line goes quiet at once, which is a
    /// click; a transmitter being stopped is not the time for a ramp.
    pub fn clear(&mut self) {
        self.queue.clear();
    }

    /// The next line sample.
    pub fn next_sample(&mut self) -> f64 {
        let n = self.sample;
        self.sample += 1;
        let t = self.symbol_time(n);
        // Symbols wholly in the past no longer contribute.
        while !self.queue.is_empty() && (self.first as f64) < t - SPAN as f64 {
            self.queue.pop_front();
            self.first += 1;
        }
        if self.queue.is_empty() {
            return 0.0;
        }
        let low = ((t - SPAN as f64).ceil() as i64).max(self.first);
        let high = ((t + SPAN as f64).floor() as i64).min(self.first + self.queue.len() as i64 - 1);
        let mut b = Complex::ZERO;
        for k in low..=high {
            b += self.queue[(k - self.first) as usize].scale(pulse_at(t - k as f64));
        }
        let turns = (self.carrier * n as f64 / FS).fract();
        let (sin, cos) = (TAU * turns).sin_cos();
        self.amplitude * (b.re * cos - b.im * sin)
    }

    /// Append `n` samples to `out`, clipped to full scale.
    pub fn fill(&mut self, n: usize, out: &mut Vec<f32>) {
        out.reserve(n);
        for _ in 0..n {
            out.push(self.next_sample().clamp(-1.0, 1.0) as f32);
        }
    }

    /// Everything queued, as samples, until the queue runs out.
    pub fn drain(&mut self) -> Vec<f32> {
        let mut out = Vec::new();
        while self.busy() {
            self.fill(1024, &mut out);
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_pulse_is_nyquist_when_squared() {
        // RRC convolved with itself is a raised cosine: one at nought and
        // nought at every other whole symbol, to within the window's taper.
        let rc = |k: f64| {
            let mut sum = 0.0;
            let dt = 1.0 / STEPS as f64;
            let mut t = -(SPAN as f64);
            while t < SPAN as f64 {
                sum += pulse_at(t) * pulse_at(t - k) * dt;
                t += dt;
            }
            sum
        };
        let peak = rc(0.0);
        for k in 1..6 {
            let isi = rc(f64::from(k)) / peak;
            assert!(isi.abs() < 0.005, "intersymbol interference {isi:.4} at {k} symbols");
        }
    }

    #[test]
    fn the_level_is_as_asked() {
        for profile in Profile::ALL {
            let mut m = Modulator::new(profile, -12.0);
            let symbols: Vec<Complex> =
                (0..4000).map(|k| if (k * 7919) % 13 < 6 { Complex::ONE } else { -Complex::ONE }).collect();
            m.push(&symbols);
            let samples = m.drain();
            let middle = &samples[samples.len() / 4..3 * samples.len() / 4];
            let rms = (middle.iter().map(|x| f64::from(*x).powi(2)).sum::<f64>() / middle.len() as f64).sqrt();
            let db = 20.0 * (rms * std::f64::consts::SQRT_2).log10();
            assert!((db + 12.0).abs() < 0.5, "{}: rms {db:.2} dBFS against -12", profile.name());
        }
    }

    #[test]
    fn silence_between_bursts() {
        let mut m = Modulator::new(Profile::Narrow, -12.0);
        let mut out = Vec::new();
        m.fill(100, &mut out);
        assert!(out.iter().all(|x| *x == 0.0));
        m.push(&[Complex::ONE; 10]);
        assert!(m.busy());
        let tail = m.drain();
        assert!(!m.busy());
        assert!(tail.iter().any(|x| x.abs() > 0.01));
    }
}
