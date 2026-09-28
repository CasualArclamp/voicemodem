//! Simulated lines, for the tests and the self test: the things a voice
//! channel does to a modem, one at a time or together.
//!
//! Each is modelled on something measured on BinModem's lines rather than
//! on a textbook: a jitter buffer's 20 ms slips, a far sound card's clock a
//! hundred-odd ppm out, a mistuned radio's carrier offset, a telephone
//! channel's band edges and group delay, G.711's companding at 8 kHz.

use std::f64::consts::{PI, TAU};

use dsp::{Complex, Resampler, bandpass};

use crate::profile::{FS, Profile};

/// A line, and what it does.
#[derive(Debug, Clone)]
pub struct Channel {
    /// Es/N0 of added white noise, in decibels; None for none.
    pub snr_db: Option<f64>,
    /// Frequency shift of the whole band, in hertz, as a mistuned single
    /// sideband radio makes; and how fast it moves, in hertz a second, as a
    /// satellite's Doppler does.
    pub shift_hz: f64,
    pub doppler_rate: f64,
    /// Slow fading between full strength and `depth` dB down, `rate` times
    /// a second: a spinning satellite, or a mobile's flutter.
    pub fade: Option<(f64, f64)>,
    /// The far clock against ours, in parts per million.
    pub ppm: f64,
    /// Gain, in decibels.
    pub gain_db: f64,
    /// A telephone channel's band: 300 to 3400 Hz, fourth-order edges, and
    /// their group delay.
    pub telephone: bool,
    /// G.711 mu-law at 8 kHz, as a VoIP call carries it.
    pub mulaw: bool,
    /// Jitter-buffer slips: at each sample position, that many samples of
    /// the last stretch played again (positive) or dropped (negative).
    pub slips: Vec<(usize, i64)>,
    pub seed: u64,
    /// The symbol rate Es/N0 is reckoned at.
    pub baud: f64,
}

impl Default for Channel {
    fn default() -> Self {
        Self { snr_db: None, shift_hz: 0.0, doppler_rate: 0.0, fade: None, ppm: 0.0, gain_db: 0.0, telephone: false, mulaw: false, slips: Vec::new(), seed: 1, baud: Profile::Wide.baud() }
    }
}

impl Channel {
    pub fn clean() -> Self {
        Self::default()
    }

    /// Reckon Es/N0 at `profile`'s symbol rate.
    pub fn profile(mut self, profile: Profile) -> Self {
        self.baud = profile.baud();
        self
    }

    pub fn noise(mut self, snr_db: f64) -> Self {
        self.snr_db = Some(snr_db);
        self
    }

    pub fn shift(mut self, hz: f64) -> Self {
        self.shift_hz = hz;
        self
    }

    /// A shift that moves by `hz_per_s` every second, from where
    /// [`Channel::shift`] puts it.
    pub fn doppler(mut self, hz_per_s: f64) -> Self {
        self.doppler_rate = hz_per_s;
        self
    }

    pub fn fading(mut self, depth_db: f64, rate_hz: f64) -> Self {
        self.fade = Some((depth_db, rate_hz));
        self
    }

    pub fn clock(mut self, ppm: f64) -> Self {
        self.ppm = ppm;
        self
    }

    pub fn gain(mut self, db: f64) -> Self {
        self.gain_db = db;
        self
    }

    pub fn telephone(mut self) -> Self {
        self.telephone = true;
        self
    }

    pub fn mulaw(mut self) -> Self {
        self.mulaw = true;
        self
    }

    pub fn slip(mut self, at: usize, samples: i64) -> Self {
        self.slips.push((at, samples));
        self
    }

    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// The line's output for `input`.
    pub fn apply(&self, input: &[f32]) -> Vec<f32> {
        let mut x: Vec<f64> = input.iter().map(|&s| f64::from(s)).collect();
        // Noise is measured against the signal as sent, before anything
        // else, so that the figure means the same whatever follows it.
        let power = {
            let busy: Vec<f64> = x.iter().copied().filter(|s| *s != 0.0).collect();
            busy.iter().map(|s| s * s).sum::<f64>() / busy.len().max(1) as f64
        };
        if self.telephone {
            let mut filter = bandpass(4, 300.0, 3400.0, FS);
            for s in &mut x {
                *s = filter.process(*s);
            }
        }
        if self.shift_hz != 0.0 || self.doppler_rate != 0.0 {
            x = shift(&x, self.shift_hz, self.doppler_rate);
        }
        if self.ppm != 0.0 {
            x = resample(&x, FS, FS * (1.0 + self.ppm * 1e-6));
        }
        if self.mulaw {
            let mut narrow = resample(&x, FS, 8000.0);
            for s in &mut narrow {
                *s = f64::from(mulaw_decode(mulaw_encode(*s)));
            }
            x = resample(&narrow, 8000.0, FS);
        }
        if !self.slips.is_empty() {
            x = slip(&x, &self.slips);
        }
        if let Some((depth, rate)) = self.fade {
            for (n, s) in x.iter_mut().enumerate() {
                let t = n as f64 / FS;
                let down = depth * (0.5 - 0.5 * (TAU * rate * t).cos());
                *s *= 10f64.powf(-down / 20.0);
            }
        }
        let gain = 10f64.powf(self.gain_db / 20.0);
        if let Some(snr) = self.snr_db {
            let sigma = (FS / 2.0 / self.baud * power * 10f64.powf(-snr / 10.0)).sqrt();
            let mut random = Random::new(self.seed);
            for s in &mut x {
                *s += sigma * random.gaussian();
            }
        }
        x.iter().map(|s| (s * gain).clamp(-1.0, 1.0) as f32).collect()
    }
}

fn resample(x: &[f64], from: f64, to: f64) -> Vec<f64> {
    let mut r = Resampler::new(from, to);
    let mut out = Vec::with_capacity((x.len() as f64 * to / from) as usize + 64);
    for &s in x.iter().chain(std::iter::repeat_n(&0.0, 64)) {
        r.process(s, &mut out);
    }
    out
}

/// Every frequency moved by `hz`, and by `rate` more every second: the
/// analytic signal, from a Hilbert transformer, turned and its real part
/// taken.
fn shift(x: &[f64], hz: f64, rate: f64) -> Vec<f64> {
    const HALF: i64 = 64;
    let taps: Vec<f64> = (-HALF..=HALF)
        .map(|n| {
            if n % 2 == 0 {
                0.0
            } else {
                let window = 0.54 + 0.46 * (PI * n as f64 / HALF as f64).cos();
                2.0 / (PI * n as f64) * window
            }
        })
        .collect();
    let at = |i: i64| usize::try_from(i).ok().and_then(|i| x.get(i)).copied().unwrap_or(0.0);
    (0..x.len() as i64)
        .map(|n| {
            let im: f64 = taps.iter().enumerate().map(|(k, h)| h * at(n - (k as i64 - HALF))).sum();
            let analytic = Complex::new(at(n), im);
            let t = n as f64 / FS;
            (analytic * Complex::from_polar(1.0, TAU * (hz * t + 0.5 * rate * t * t))).re
        })
        .collect()
}

/// The slips, in order of position, applied as a jitter buffer would: a
/// repeat of the stretch just played, or a stretch skipped.
fn slip(x: &[f64], slips: &[(usize, i64)]) -> Vec<f64> {
    let mut sorted = slips.to_vec();
    sorted.sort_by_key(|s| s.0);
    let mut out = Vec::with_capacity(x.len());
    let mut from = 0usize;
    for (at, by) in sorted {
        let at = at.min(x.len()).max(from);
        out.extend_from_slice(&x[from..at]);
        if by > 0 {
            let n = by as usize;
            let start = out.len().saturating_sub(n);
            let repeat: Vec<f64> = out[start..].to_vec();
            out.extend(repeat);
            from = at;
        } else {
            from = (at + by.unsigned_abs() as usize).min(x.len());
        }
    }
    out.extend_from_slice(&x[from..]);
    out
}

/// G.711 mu-law, from a sample at full scale one.
pub fn mulaw_encode(x: f64) -> u8 {
    const BIAS: i32 = 0x84;
    const CLIP: i32 = 32_635;
    let linear = (x * 32_767.0).round().clamp(-32_768.0, 32_767.0) as i32;
    let sign = if linear < 0 { 0x80 } else { 0 };
    let magnitude = linear.abs().min(CLIP) + BIAS;
    let exponent = (7 - (magnitude << 17).leading_zeros().min(7)) as i32;
    let mantissa = (magnitude >> (exponent + 3)) & 0x0F;
    !((sign | (exponent << 4) | mantissa) as u8)
}

/// G.711 mu-law, to a sample at full scale one.
pub fn mulaw_decode(code: u8) -> f32 {
    const BIAS: i32 = 0x84;
    let u = !code;
    let exponent = i32::from((u >> 4) & 0x07);
    let mantissa = i32::from(u & 0x0F);
    let magnitude = (((mantissa << 3) + BIAS) << exponent) - BIAS;
    let linear = if u & 0x80 != 0 { -magnitude } else { magnitude };
    linear as f32 / 32_768.0
}

/// A small, fast, reproducible generator.
#[derive(Debug, Clone)]
pub struct Random(u64);

impl Random {
    pub fn new(seed: u64) -> Self {
        Self(seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1)
    }

    pub fn next_u64(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }

    pub fn uniform(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    }

    pub fn gaussian(&mut self) -> f64 {
        let (a, b) = (self.uniform(), self.uniform());
        (-2.0 * a.ln()).sqrt() * (TAU * b).cos()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mulaw_round_trips_within_its_step() {
        for i in -1000..=1000 {
            let x = f64::from(i) / 1000.0 * 0.9;
            let y = f64::from(mulaw_decode(mulaw_encode(x)));
            // Steps are about 3% of the level, plus the smallest step.
            assert!((x - y).abs() <= 0.04 * x.abs() + 1e-3, "{x} came back {y}");
        }
    }

    #[test]
    fn a_shift_moves_a_tone() {
        let tone: Vec<f64> = (0..16_000).map(|n| (TAU * 1000.0 * f64::from(n) / FS).sin()).collect();
        let moved = shift(&tone, 50.0, 0.0);
        // Correlate against 1050 Hz and 950 Hz over the middle.
        let power = |hz: f64| {
            let c = (4000..12_000).fold(Complex::ZERO, |sum, n| {
                sum + Complex::from_polar(moved[n], -TAU * hz * n as f64 / FS)
            });
            c.norm_sqr()
        };
        assert!(power(1050.0) > 1000.0 * power(950.0));
    }

    #[test]
    fn slips_insert_and_drop() {
        let x: Vec<f64> = (0..100).map(f64::from).collect();
        let y = slip(&x, &[(50, 10)]);
        assert_eq!(y.len(), 110);
        assert_eq!(y[50], 40.0);
        assert_eq!(y[60], 50.0);
        let z = slip(&x, &[(50, -10)]);
        assert_eq!(z.len(), 90);
        assert_eq!(z[50], 60.0);
    }
}
