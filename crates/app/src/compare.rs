//! Two recordings of the same speech, set against each other: how late the
//! second is, how much louder, and how far its spectrum strays.
//!
//! For judging a codec, or a path, without listening: a codec that has made
//! a mess reads as a large log-spectral distance and an envelope that no
//! longer follows the original's. The figures are the standard ones:
//!
//! - **delay**: where the two log-energy envelopes, 10 ms a point, line up
//!   best;
//! - **gain**: the second's level against the first's over the speech;
//! - **log-spectral distance**: over 32 ms frames where the first has speech,
//!   the rms difference in dB between the two power spectra from 100 Hz to
//!   3.8 kHz, the second's level matched first;
//! - **envelope correlation**: of the two log-energy envelopes, aligned.

use dsp::Fft;
use modem::profile::FS;

/// Samples a 10 ms envelope point, a spectral frame, and its hop.
const POINT: usize = 160;
const FRAME: usize = 512;
const HOP: usize = 160;

/// What a comparison found.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Comparison {
    pub delay_ms: f64,
    pub gain_db: f64,
    pub lsd_db: f64,
    pub envelope: f64,
    pub frames: usize,
}

fn envelope(x: &[f32]) -> Vec<f64> {
    x.chunks(POINT)
        .map(|c| 10.0 * (c.iter().map(|s| f64::from(*s).powi(2)).sum::<f64>() / c.len() as f64 + 1e-10).log10())
        .collect()
}

fn correlation(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len().min(b.len());
    if n < 2 {
        return 0.0;
    }
    let (ma, mb) = (a[..n].iter().sum::<f64>() / n as f64, b[..n].iter().sum::<f64>() / n as f64);
    let (mut ab, mut aa, mut bb) = (0.0, 0.0, 0.0);
    for i in 0..n {
        let (x, y) = (a[i] - ma, b[i] - mb);
        ab += x * y;
        aa += x * x;
        bb += y * y;
    }
    ab / (aa * bb).sqrt().max(1e-12)
}

/// Power spectra of every frame of `x` starting at `from`, bins `low` to
/// `high`, in dB.
fn spectra(x: &[f32], starts: &[usize], low: usize, high: usize) -> Vec<Vec<f64>> {
    let fft = Fft::new(FRAME);
    let window: Vec<f64> =
        (0..FRAME).map(|i| 0.5 - 0.5 * (std::f64::consts::TAU * i as f64 / FRAME as f64).cos()).collect();
    starts
        .iter()
        .map(|&s| {
            let mut re: Vec<f64> = (0..FRAME).map(|i| f64::from(x.get(s + i).copied().unwrap_or(0.0)) * window[i]).collect();
            let mut im = vec![0.0; FRAME];
            fft.process(&mut re, &mut im);
            (low..high).map(|k| 10.0 * (re[k] * re[k] + im[k] * im[k] + 1e-12).log10()).collect()
        })
        .collect()
}

/// `b` against `a`, both at the modem's rate.
pub fn compare(a: &[f32], b: &[f32]) -> Comparison {
    let (ea, eb) = (envelope(a), envelope(b));
    // Where the envelopes line up: b lagging a by `lag` points.
    let most = (3.0 * FS / POINT as f64) as i64;
    let score = |lag: i64| {
        let (x, y): (&[f64], &[f64]) = if lag >= 0 {
            let l = lag as usize;
            (&ea[..ea.len().min(eb.len().saturating_sub(l))], eb.get(l..).unwrap_or(&[]))
        } else {
            let l = (-lag) as usize;
            (ea.get(l..).unwrap_or(&[]), &eb[..eb.len().min(ea.len().saturating_sub(l))])
        };
        correlation(x, y)
    };
    let (lag, envelope) = (-most..=most).map(|l| (l, score(l))).max_by(|p, q| p.1.total_cmp(&q.1)).unwrap_or((0, 0.0));
    let shift = lag * POINT as i64;

    // Frames where a has speech: within 30 dB of its loudest.
    let loudest = ea.iter().copied().fold(f64::MIN, f64::max);
    let starts: Vec<usize> = (0..a.len().saturating_sub(FRAME))
        .step_by(HOP)
        .filter(|&s| {
            let e = ea.get(s / POINT).copied().unwrap_or(-100.0);
            let t = s as i64 + shift;
            e > loudest - 30.0 && t >= 0 && (t as usize) + FRAME <= b.len()
        })
        .collect();
    let shifted: Vec<usize> = starts.iter().map(|&s| (s as i64 + shift) as usize).collect();
    let bin = FS / FRAME as f64;
    let (low, high) = ((100.0 / bin).round() as usize, (3800.0 / bin).round() as usize);
    let (sa, sb) = (spectra(a, &starts, low, high), spectra(b, &shifted, low, high));
    let frames = sa.len();
    let mean = |s: &[Vec<f64>]| s.iter().flatten().sum::<f64>() / (frames * (high - low)).max(1) as f64;
    let gain_db = mean(&sb) - mean(&sa);
    let lsd_db = sa
        .iter()
        .zip(&sb)
        .map(|(x, y)| (x.iter().zip(y).map(|(p, q)| (q - gain_db - p).powi(2)).sum::<f64>() / x.len() as f64).sqrt())
        .sum::<f64>()
        / frames.max(1) as f64;
    Comparison { delay_ms: shift as f64 / FS * 1000.0, gain_db, lsd_db, envelope, frames }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_delayed_quieter_copy_is_found_as_one() {
        let a = voice::synthetic_speech(4.0, FS);
        let mut b = vec![0.0f32; 4000];
        b.extend(a.iter().map(|x| x * 0.5));
        let c = compare(&a, &b);
        assert!((c.delay_ms - 250.0).abs() < 11.0, "delay {}", c.delay_ms);
        assert!((c.gain_db + 6.0).abs() < 0.5, "gain {}", c.gain_db);
        assert!(c.lsd_db < 1.0, "lsd {}", c.lsd_db);
        assert!(c.envelope > 0.99);
    }

    #[test]
    fn noise_is_nothing_like_speech() {
        let a = voice::synthetic_speech(4.0, FS);
        let mut random = modem::channel::Random::new(5);
        let b: Vec<f32> = (0..a.len()).map(|_| (0.1 * random.gaussian()) as f32).collect();
        let c = compare(&a, &b);
        assert!(c.envelope < 0.5 && c.lsd_db > 8.0, "{c:?}");
    }
}
