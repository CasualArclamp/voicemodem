//! Something speech-like to test with, when there is no recording to hand.
//!
//! A glottal buzz whose pitch wanders between 95 and 150 Hz, through two
//! formants that glide between the vowels a, e, i, o and u, cut into
//! syllables four times a second with a pause every couple of seconds. It is
//! not speech, but it has what Codec 2 models -- a pitch, a spectral
//! envelope, voicing that starts and stops -- so a codec that mangles it
//! would mangle speech.

use std::f64::consts::TAU;

/// Formant pairs, in hertz, of five vowels.
const VOWELS: [(f64, f64); 5] = [(730.0, 1090.0), (530.0, 1840.0), (270.0, 2290.0), (570.0, 840.0), (300.0, 870.0)];

/// `seconds` of synthetic speech at `rate` samples a second, peaking near
/// -6 dBFS.
pub fn synthetic_speech(seconds: f64, rate: f64) -> Vec<f32> {
    let n = (seconds * rate) as usize;
    let mut phase = 0.0f64;
    (0..n)
        .map(|i| {
            let t = i as f64 / rate;
            let pitch = 122.0 + 27.0 * (TAU * 0.31 * t).sin() + 8.0 * (TAU * 1.7 * t).sin();
            phase = (phase + pitch / rate).fract();
            // Which vowel, gliding into the next over each syllable.
            let syllable = t * 4.0;
            let (a, b) = (VOWELS[syllable as usize % 5], VOWELS[(syllable as usize + 1) % 5]);
            let glide = syllable.fract();
            let f1 = a.0 + (b.0 - a.0) * glide;
            let f2 = a.1 + (b.1 - a.1) * glide;
            // The harmonics of the buzz, shaped by the two formants.
            let mut x = 0.0;
            let mut h = 1.0;
            while h * pitch < 3600.0 {
                let f = h * pitch;
                let gain = (-((f - f1) / 110.0).powi(2)).exp() + 0.5 * (-((f - f2) / 160.0).powi(2)).exp() + 0.02;
                x += gain * (TAU * h * phase).sin() / h.sqrt();
                h += 1.0;
            }
            // Syllables, and a breath every two and a half seconds.
            let envelope = (std::f64::consts::PI * glide).sin().powf(0.6);
            let breath = if t % 2.5 > 2.1 { 0.0 } else { 1.0 };
            (0.18 * envelope * breath * x) as f32
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_is_loud_enough_and_never_clips() {
        let x = synthetic_speech(3.0, 8000.0);
        let peak = x.iter().fold(0.0f32, |m, s| m.max(s.abs()));
        assert!(peak > 0.2 && peak < 1.0, "peak {peak}");
    }
}
