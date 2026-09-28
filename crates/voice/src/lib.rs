//! Speech in and out of the modem: the codecs, and the plumbing either side
//! of them.
//!
//! - [`SpeechCodec`] is a codec as a stream: speech in, frames out as they
//!   are ready; frames in, speech out. Codec 2 makes a frame for every 20 or
//!   40 ms it is given; a neural codec may want a chunk at a time. Neither
//!   the talker nor the listener needs to know which.
//! - [`Talker`] turns speech at the line's rate into codec frames.
//! - [`Listener`] turns what the receiver hears back into speech, with
//!   silence in place of what was lost, and [`Playout`] meters it out to a
//!   sound card that takes a little at a time.

mod codec2;
#[cfg(feature = "neural")]
pub mod encodec;
mod synthetic;

use std::collections::VecDeque;

use dsp::Resampler;
use modem::{Codec, VoiceCodeword, VoiceMode};

pub use crate::codec2::Codec2Speech;
pub use crate::synthetic::synthetic_speech;

/// A speech codec as a stream.
pub trait SpeechCodec: Send + std::fmt::Debug {
    fn codec(&self) -> Codec;

    /// Speech samples a second the codec works at.
    fn sample_rate(&self) -> u32;

    /// Samples of speech, at [`SpeechCodec::sample_rate`], each frame is made
    /// of, in order.
    fn frame_samples(&self) -> usize;

    /// Take speech, at [`SpeechCodec::sample_rate`], full scale one; push
    /// each frame that is complete, as its bits.
    fn encode(&mut self, speech: &[f32], frames: &mut Vec<Vec<u8>>);

    /// Take a frame's bits, or None for a frame that never arrived; push the
    /// speech it makes.
    fn decode(&mut self, frame: Option<&[u8]>, speech: &mut Vec<f32>);

    /// The speaker has stopped: push frames for whatever speech is still
    /// held back, made up with silence.
    fn finish_encoding(&mut self, _frames: &mut Vec<Vec<u8>>) {}

    /// The transmission is over: push the speech for whatever frames are
    /// still held back.
    fn finish_decoding(&mut self, _speech: &mut Vec<f32>) {}
}

/// A codec ready to use, or why it is not.
pub fn open(codec: Codec) -> Result<Box<dyn SpeechCodec>, String> {
    match codec {
        Codec::Encodec1500 | Codec::Encodec3000 => open_neural(codec),
        _ => Ok(Box::new(Codec2Speech::new(codec)?)),
    }
}

#[cfg(feature = "neural")]
fn open_neural(codec: Codec) -> Result<Box<dyn SpeechCodec>, String> {
    Ok(Box::new(encodec::EncodecSpeech::new(codec)?))
}

#[cfg(not(feature = "neural"))]
fn open_neural(codec: Codec) -> Result<Box<dyn SpeechCodec>, String> {
    Err(format!("{} is not built into this copy of the program", codec.label()))
}

/// Whether `codec` can be opened here, without opening it: for a neural
/// codec, whether its weights are where they are looked for; if not, what to
/// do about it. Cheap enough to ask whenever a list of modes is shown.
pub fn available(codec: Codec) -> Result<(), String> {
    match codec {
        Codec::Encodec1500 | Codec::Encodec3000 => neural_available(),
        _ => Ok(()),
    }
}

#[cfg(feature = "neural")]
fn neural_available() -> Result<(), String> {
    encodec::find_weights().map(|_| ())
}

#[cfg(not(feature = "neural"))]
fn neural_available() -> Result<(), String> {
    Err("EnCodec is not built into this copy of the program".into())
}

/// Sample rate conversion over runs of samples.
#[derive(Debug, Clone)]
pub struct Rate {
    resampler: Option<Resampler>,
    out: Vec<f64>,
}

impl Rate {
    pub fn new(from: f64, to: f64) -> Self {
        let resampler = ((from - to).abs() > 1e-9).then(|| Resampler::new(from, to));
        Self { resampler, out: Vec::new() }
    }

    /// Convert `input`, appending to `output`.
    pub fn process(&mut self, input: &[f32], output: &mut Vec<f32>) {
        match &mut self.resampler {
            None => output.extend_from_slice(input),
            Some(r) => {
                self.out.clear();
                for &x in input {
                    r.process(f64::from(x), &mut self.out);
                }
                output.extend(self.out.iter().map(|&y| y as f32));
            }
        }
    }
}

/// A codec frame, and whether the speech it came from was silence -- a pause
/// a transmitter that has fallen behind may drop to catch up.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Spoken {
    pub bits: Vec<u8>,
    pub quiet: bool,
}

/// How far under the loudest recent speech a frame counts as silence, and
/// the level under which it always does.
const QUIET_BELOW_LOUDEST_DB: f64 = 30.0;
const QUIET_FLOOR_DB: f64 = -60.0;
/// How fast the loudest recent speech is forgotten, in decibels a frame.
const LOUDEST_DECAY_DB: f64 = 0.05;

/// Speech at the line's rate in, codec frames out.
#[derive(Debug)]
pub struct Talker {
    codec: Box<dyn SpeechCodec>,
    rate: Rate,
    converted: Vec<f32>,
    /// Speech at the codec's rate not yet a whole frame, the levels of the
    /// frames the codec has not yet handed back, and the loudest recent.
    window: Vec<f32>,
    levels: VecDeque<f64>,
    loudest: f64,
    raw: Vec<Vec<u8>>,
}

impl Talker {
    /// A talker for `mode`, taking speech at `line_rate`.
    pub fn new(mode: &VoiceMode, line_rate: f64) -> Result<Self, String> {
        let codec = open(mode.codec)?;
        let rate = Rate::new(line_rate, f64::from(codec.sample_rate()));
        Ok(Self {
            codec,
            rate,
            converted: Vec::new(),
            window: Vec::new(),
            levels: VecDeque::new(),
            loudest: QUIET_FLOOR_DB,
            raw: Vec::new(),
        })
    }

    /// Take speech; push the frames it completes.
    pub fn speak(&mut self, speech: &[f32], frames: &mut Vec<Spoken>) {
        self.converted.clear();
        self.rate.process(speech, &mut self.converted);
        // Each frame's level, from the same samples the codec makes it of.
        let per = self.codec.frame_samples();
        for &x in &self.converted {
            self.window.push(x);
            if self.window.len() == per {
                let power = self.window.iter().map(|s| f64::from(*s).powi(2)).sum::<f64>() / per as f64;
                self.levels.push_back(10.0 * (power + 1e-12).log10());
                self.window.clear();
            }
        }
        self.raw.clear();
        self.codec.encode(&self.converted, &mut self.raw);
        self.hand_back(frames);
    }

    /// The speaker has let go: push frames for whatever is held back.
    pub fn finish(&mut self, frames: &mut Vec<Spoken>) {
        self.raw.clear();
        self.codec.finish_encoding(&mut self.raw);
        self.hand_back(frames);
        self.window.clear();
        self.levels.clear();
    }

    fn hand_back(&mut self, frames: &mut Vec<Spoken>) {
        for bits in self.raw.drain(..) {
            // A frame made up of padding at the end has no level of its own:
            // it is not dropped.
            let quiet = match self.levels.pop_front() {
                Some(db) => {
                    self.loudest = db.max(self.loudest - LOUDEST_DECAY_DB);
                    db < (self.loudest - QUIET_BELOW_LOUDEST_DB).max(QUIET_FLOOR_DB)
                }
                None => false,
            };
            frames.push(Spoken { bits, quiet });
        }
    }
}

/// What the receiver hears, back to speech at the line's rate.
#[derive(Debug)]
pub struct Listener {
    mode: &'static VoiceMode,
    codec: Box<dyn SpeechCodec>,
    rate: Rate,
    decoded: Vec<f32>,
    /// Codewords heard, and lost.
    pub heard: usize,
    pub lost: usize,
    /// The transmission's text, as it has come.
    text: VecDeque<u8>,
}

/// Characters of a transmission's text kept.
const TEXT: usize = 64;

impl Listener {
    /// A listener for `mode`, giving speech at `line_rate`.
    pub fn new(mode: &'static VoiceMode, line_rate: f64) -> Result<Self, String> {
        let codec = open(mode.codec)?;
        let rate = Rate::new(f64::from(codec.sample_rate()), line_rate);
        Ok(Self { mode, codec, rate, decoded: Vec::new(), heard: 0, lost: 0, text: VecDeque::new() })
    }

    pub fn mode(&self) -> &'static VoiceMode {
        self.mode
    }

    /// A codeword as the receiver had it -- None if it did not decode --
    /// turned into speech appended to `speech`.
    pub fn hear(&mut self, codeword: Option<&VoiceCodeword>, speech: &mut Vec<f32>) {
        self.decoded.clear();
        match codeword {
            Some(c) => {
                self.heard += 1;
                for frame in &c.frames {
                    self.codec.decode(Some(frame), &mut self.decoded);
                }
                if c.end {
                    self.codec.finish_decoding(&mut self.decoded);
                }
                if c.text != 0 {
                    self.text.push_back(c.text);
                    while self.text.len() > TEXT {
                        self.text.pop_front();
                    }
                }
            }
            None => {
                // As much silence as a codeword carries on average: its air
                // time, since the talker sends speech as fast as it comes.
                self.lost += 1;
                let frames = (self.mode.codeword_seconds() / self.mode.codec.frame_seconds()).round() as usize;
                for _ in 0..frames {
                    self.codec.decode(None, &mut self.decoded);
                }
            }
        }
        self.rate.process(&self.decoded, speech);
    }

    /// The text heard so far, most recent last.
    pub fn text(&self) -> String {
        self.text.iter().map(|&b| char::from(b)).collect()
    }
}

/// Speech metered out to a sound card a little at a time, after enough has
/// built up to ride out the gaps between codewords.
#[derive(Debug, Clone)]
pub struct Playout {
    queue: VecDeque<f32>,
    /// Samples to gather before playing, and whether that has happened.
    preroll: usize,
    playing: bool,
    /// Samples of silence played because nothing was there.
    pub starved: u64,
}

impl Playout {
    /// A playout that waits for `preroll` samples before it starts.
    pub fn new(preroll: usize) -> Self {
        Self { queue: VecDeque::new(), preroll, playing: false, starved: 0 }
    }

    pub fn push(&mut self, speech: &[f32]) {
        self.queue.extend(speech.iter().copied());
    }

    /// Samples waiting.
    pub fn len(&self) -> usize {
        self.queue.len()
    }

    pub fn is_empty(&self) -> bool {
        self.queue.is_empty()
    }

    /// The next `n` samples: speech if there is enough, silence if not.
    pub fn pull(&mut self, n: usize, out: &mut Vec<f32>) {
        if !self.playing && self.queue.len() >= self.preroll {
            self.playing = true;
        }
        if !self.playing {
            out.extend(std::iter::repeat_n(0.0, n));
            return;
        }
        for _ in 0..n {
            match self.queue.pop_front() {
                Some(x) => out.push(x),
                None => {
                    // Ran dry: gather again before playing, rather than
                    // stuttering sample by sample.
                    self.playing = false;
                    self.starved += 1;
                    out.push(0.0);
                }
            }
        }
    }

    /// Drop everything waiting: a new transmission starts from nothing.
    pub fn clear(&mut self) {
        self.queue.clear();
        self.playing = false;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_codec2_mode_opens() {
        for codec in Codec::ALL {
            if !matches!(codec, Codec::Encodec1500 | Codec::Encodec3000) {
                assert_eq!(open(codec).map(|c| c.codec()), Ok(codec));
            }
        }
    }

    #[cfg(feature = "neural")]
    #[test]
    fn encodec_opens_or_says_where_its_weights_go() {
        match open(Codec::Encodec1500) {
            Ok(c) => assert_eq!(c.codec(), Codec::Encodec1500),
            Err(e) => assert!(e.contains("huggingface.co/facebook/encodec_24khz"), "{e}"),
        }
    }

    #[test]
    fn playout_waits_then_plays_then_waits_again() {
        let mut p = Playout::new(4);
        let mut out = Vec::new();
        p.push(&[1.0, 2.0]);
        p.pull(2, &mut out);
        assert_eq!(out, vec![0.0, 0.0]);
        p.push(&[3.0, 4.0]);
        out.clear();
        p.pull(5, &mut out);
        assert_eq!(out, vec![1.0, 2.0, 3.0, 4.0, 0.0]);
        assert_eq!(p.starved, 1);
    }
}
