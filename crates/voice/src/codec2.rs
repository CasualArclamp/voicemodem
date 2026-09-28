//! Codec 2, at 3200 down to 1200 bit/s.

use codec2::{Codec2, Codec2Mode};
use modem::Codec;

use crate::SpeechCodec;

/// Codec 2 as a [`SpeechCodec`]: 8 kHz speech, a frame every 20 or 40 ms.
pub struct Codec2Speech {
    codec: Codec,
    inner: Codec2,
    /// Speech waiting for a whole frame, as the codec's 16-bit samples.
    pending: Vec<i16>,
    samples: usize,
    bits: usize,
    packed: Vec<u8>,
    out: Vec<i16>,
}

impl std::fmt::Debug for Codec2Speech {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Codec2Speech").field("codec", &self.codec).finish_non_exhaustive()
    }
}

impl Codec2Speech {
    pub fn new(codec: Codec) -> Result<Self, String> {
        let mode = match codec {
            Codec::Codec2_3200 => Codec2Mode::MODE_3200,
            Codec::Codec2_2400 => Codec2Mode::MODE_2400,
            Codec::Codec2_1600 => Codec2Mode::MODE_1600,
            Codec::Codec2_1400 => Codec2Mode::MODE_1400,
            Codec::Codec2_1300 => Codec2Mode::MODE_1300,
            Codec::Codec2_1200 => Codec2Mode::MODE_1200,
            other => return Err(format!("{} is not Codec 2", other.label())),
        };
        let inner = Codec2::new(mode);
        let (samples, bits) = (inner.samples_per_frame(), inner.bits_per_frame());
        if bits != codec.bits() {
            return Err(format!("{} frames are {bits} bits, not {}", codec.label(), codec.bits()));
        }
        Ok(Self { codec, inner, pending: Vec::new(), samples, bits, packed: vec![0; bits.div_ceil(8)], out: vec![0; samples] })
    }
}

impl SpeechCodec for Codec2Speech {
    fn codec(&self) -> Codec {
        self.codec
    }

    fn sample_rate(&self) -> u32 {
        8000
    }

    fn encode(&mut self, speech: &[f32], frames: &mut Vec<Vec<u8>>) {
        self.pending.extend(speech.iter().map(|&x| (x * 32_767.0).round().clamp(-32_768.0, 32_767.0) as i16));
        let mut at = 0;
        while self.pending.len() - at >= self.samples {
            self.inner.encode(&mut self.packed, &self.pending[at..at + self.samples]);
            // Codec 2 packs its bits most significant first.
            frames.push((0..self.bits).map(|i| (self.packed[i / 8] >> (7 - i % 8)) & 1).collect());
            at += self.samples;
        }
        self.pending.drain(..at);
    }

    fn decode(&mut self, frame: Option<&[u8]>, speech: &mut Vec<f32>) {
        match frame {
            Some(bits) => {
                self.packed.fill(0);
                for (i, &b) in bits.iter().take(self.bits).enumerate() {
                    self.packed[i / 8] |= (b & 1) << (7 - i % 8);
                }
                self.inner.decode(&mut self.out, &self.packed);
                speech.extend(self.out.iter().map(|&s| f32::from(s) / 32_768.0));
            }
            None => speech.extend(std::iter::repeat_n(0.0, self.samples)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::TAU;

    /// A second of a vowel-like sound: a 120 Hz buzz through a couple of
    /// formant-ish resonances, rising and falling in level.
    fn vowel() -> Vec<f32> {
        (0..8000)
            .map(|n| {
                let t = n as f32 / 8000.0;
                let envelope = 0.3 * (TAU * 1.5 * t).sin().abs();
                let mut x = 0.0;
                for h in 1..30 {
                    let f = 120.0 * h as f32;
                    let formant = (-((f - 700.0) / 150.0).powi(2)).exp() + 0.6 * (-((f - 1200.0) / 200.0).powi(2)).exp();
                    x += formant * (TAU * f * t).sin();
                }
                envelope * x
            })
            .collect()
    }

    #[test]
    fn frames_come_out_the_size_the_modem_expects() {
        for codec in [Codec::Codec2_3200, Codec::Codec2_1600, Codec::Codec2_1200] {
            let mut c = Codec2Speech::new(codec).unwrap();
            let mut frames = Vec::new();
            c.encode(&vowel(), &mut frames);
            let expected = (1.0 / codec.frame_seconds()).round() as usize;
            assert_eq!(frames.len(), expected, "{}", codec.label());
            assert!(frames.iter().all(|f| f.len() == codec.bits()));
        }
    }

    #[test]
    fn speech_survives_the_round_trip() {
        let speech = vowel();
        let mut c = Codec2Speech::new(Codec::Codec2_1200).unwrap();
        let mut frames = Vec::new();
        c.encode(&speech, &mut frames);
        let mut back = Vec::new();
        for f in &frames {
            c.decode(Some(f), &mut back);
        }
        assert_eq!(back.len(), speech.len());
        let power = |x: &[f32]| x.iter().map(|s| s * s).sum::<f32>() / x.len() as f32;
        let ratio = power(&back) / power(&speech);
        assert!((0.1..10.0).contains(&ratio), "speech came back at {:.1} dB", 10.0 * ratio.log10());
    }
}
