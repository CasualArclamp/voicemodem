//! Meta's EnCodec, 24 kHz model, run by candle: 1.5 kbit/s on two
//! codebooks, 3 kbit/s on four, a frame every 13 1/3 ms.
//!
//! The model is not a stream. Candle's implementation takes a whole clip
//! and gives back all its codes, and gives a clip's worth of audio back for
//! them. So it is run on a chunk at a time -- [`CHUNK`] frames -- with the
//! [`CONTEXT`] frames before the chunk run again ahead of it, and only the
//! chunk's own frames kept. The 24 kHz model's convolutions are causal, so a
//! frame's codes depend only on what came before it; with enough of that in
//! front, a chunk comes out as it would have in one long clip, and the
//! joins do not click.
//!
//! The weights are not part of this program. They are Meta's, under
//! CC-BY-NC 4.0 -- fine for amateur radio, which is non-commercial by
//! definition, but not ours to redistribute under the GPL. They are looked
//! for, in order, at `%VOICEMODEM_ENCODEC%`, as `encodec_24khz.safetensors`
//! beside the program, and in `%APPDATA%\voicemodem\`. The file is
//! `model.safetensors` from huggingface.co/facebook/encodec_24khz, about
//! 93 MB, renamed.

use std::path::PathBuf;
use std::sync::{Arc, Mutex, OnceLock};

use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::encodec::{Config, Model};
use modem::Codec;

use crate::SpeechCodec;

/// Samples a frame at 24 kHz.
pub const HOP: usize = 320;
/// Frames run through the model at a time: 200 ms.
pub const CHUNK: usize = 15;
/// Frames of what came before run again ahead of each chunk: 400 ms.
pub const CONTEXT: usize = 30;
/// Bits a codebook index takes: 1024 entries.
const INDEX_BITS: usize = 10;

/// The file name the weights are looked for under.
pub const WEIGHTS: &str = "encodec_24khz.safetensors";

/// What a chunk is run through: the model itself, or something standing in
/// for it in a test.
pub trait Network: Send + Sync + std::fmt::Debug {
    /// Codes for `audio` at 24 kHz: for each frame, each codebook's index.
    fn encode(&self, audio: &[f32], codebooks: usize) -> Result<Vec<Vec<u32>>, String>;
    /// Audio at 24 kHz for frames of codes.
    fn decode(&self, codes: &[Vec<u32>]) -> Result<Vec<f32>, String>;
}

/// The model, loaded once and shared by every encoder and decoder.
#[derive(Debug)]
pub struct Encodec {
    model: Model,
}

fn fail(e: candle_core::Error) -> String {
    e.to_string()
}

impl Network for Encodec {
    fn encode(&self, audio: &[f32], codebooks: usize) -> Result<Vec<Vec<u32>>, String> {
        let input = Tensor::from_slice(audio, (1, 1, audio.len()), &Device::Cpu).map_err(fail)?;
        let codes = self.model.encode(&input).map_err(fail)?;
        // One clip, the first `codebooks` of the quantiser's layers.
        let codes = codes.i((0, ..codebooks, ..)).map_err(fail)?.t().map_err(fail)?;
        codes.to_dtype(DType::U32).map_err(fail)?.to_vec2::<u32>().map_err(fail)
    }

    fn decode(&self, codes: &[Vec<u32>]) -> Result<Vec<f32>, String> {
        let (frames, books) = (codes.len(), codes.first().map_or(0, Vec::len));
        let flat: Vec<u32> = codes.iter().flatten().copied().collect();
        let input = Tensor::from_vec(flat, (1, frames, books), &Device::Cpu)
            .and_then(|t| t.transpose(1, 2))
            .and_then(|t| t.contiguous())
            .map_err(fail)?;
        let audio = self.model.decode(&input).map_err(fail)?;
        audio.flatten_all().and_then(|a| a.to_dtype(DType::F32)).and_then(|a| a.to_vec1::<f32>()).map_err(fail)
    }
}

/// Where the weights might be, in the order they are looked for.
pub fn weight_paths() -> Vec<PathBuf> {
    let mut paths = Vec::new();
    if let Some(p) = std::env::var_os("VOICEMODEM_ENCODEC") {
        paths.push(PathBuf::from(p));
    }
    if let Some(dir) = std::env::current_exe().ok().and_then(|e| e.parent().map(PathBuf::from)) {
        paths.push(dir.join(WEIGHTS));
    }
    if let Some(appdata) = std::env::var_os("APPDATA") {
        paths.push(PathBuf::from(appdata).join("voicemodem").join(WEIGHTS));
    }
    paths
}

/// The model, loading it the first time it is asked for.
pub fn shared() -> Result<Arc<dyn Network>, String> {
    static LOADED: OnceLock<Mutex<Option<Arc<Encodec>>>> = OnceLock::new();
    let slot = LOADED.get_or_init(|| Mutex::new(None));
    let mut guard = slot.lock().map_err(|_| "the EnCodec model's lock is poisoned".to_string())?;
    if let Some(model) = guard.as_ref() {
        return Ok(model.clone());
    }
    let paths = weight_paths();
    let Some(path) = paths.iter().find(|p| p.is_file()) else {
        let tried: Vec<String> = paths.iter().map(|p| p.display().to_string()).collect();
        return Err(format!(
            "EnCodec needs Meta's weights, which are not part of this program (CC-BY-NC 4.0): download \
             model.safetensors from https://huggingface.co/facebook/encodec_24khz (about 93 MB) and save it \
             as one of: {}",
            tried.join(", ")
        ));
    };
    let bytes = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let vb = VarBuilder::from_buffered_safetensors(bytes, DType::F32, &Device::Cpu).map_err(fail)?;
    let model = Model::new(&Config::default(), vb).map_err(|e| format!("{}: {e}", path.display()))?;
    let model = Arc::new(Encodec { model });
    *guard = Some(model.clone());
    Ok(model)
}

/// EnCodec as a [`SpeechCodec`].
#[derive(Debug)]
pub struct EncodecSpeech {
    codec: Codec,
    network: Arc<dyn Network>,
    codebooks: usize,
    /// Speech not yet encoded, and the audio of the frames before it.
    input: Vec<f32>,
    heard_before: Vec<f32>,
    /// Frames not yet decoded, and the codes of the frames before them.
    pending: Vec<Vec<u32>>,
    decoded_before: Vec<Vec<u32>>,
}

impl EncodecSpeech {
    pub fn new(codec: Codec) -> Result<Self, String> {
        Self::with_network(codec, shared()?)
    }

    /// An EnCodec speech codec running `network`.
    pub fn with_network(codec: Codec, network: Arc<dyn Network>) -> Result<Self, String> {
        let codebooks = match codec {
            Codec::Encodec1500 => 2,
            Codec::Encodec3000 => 4,
            other => return Err(format!("{} is not EnCodec", other.label())),
        };
        debug_assert_eq!(codebooks * INDEX_BITS, codec.bits());
        Ok(Self {
            codec,
            network,
            codebooks,
            input: Vec::new(),
            heard_before: Vec::new(),
            pending: Vec::new(),
            decoded_before: Vec::new(),
        })
    }

    fn encode_chunk(&mut self, chunk: &[f32], frames: &mut Vec<Vec<u8>>) {
        let mut audio = self.heard_before.clone();
        audio.extend_from_slice(chunk);
        let wanted = chunk.len() / HOP;
        // The chunk's frames counted from the start, after the context's:
        // anything the model makes past them is padding, not speech.
        let from = self.heard_before.len() / HOP;
        match self.network.encode(&audio, self.codebooks) {
            Ok(codes) if codes.len() >= from + wanted => {
                for code in &codes[from..from + wanted] {
                    frames.push(pack(code));
                }
            }
            // A model that fails sends silence rather than nothing, so that
            // the speech after it stays in time.
            _ => frames.extend(std::iter::repeat_n(vec![0; self.codec.bits()], wanted)),
        }
        let keep = audio.len().min(CONTEXT * HOP);
        self.heard_before = audio[audio.len() - keep..].to_vec();
    }

    fn decode_pending(&mut self, speech: &mut Vec<f32>) {
        if self.pending.is_empty() {
            return;
        }
        let new = self.pending.len();
        let mut codes = std::mem::take(&mut self.decoded_before);
        let from = codes.len() * HOP;
        codes.append(&mut self.pending);
        // The new frames' audio counted from the start, after the context's.
        // Candle's decoder gives back a few hundred samples more than the
        // frames account for, past their end; taken from the end, every
        // chunk came out 370 samples early with some of that tail in it.
        match self.network.decode(&codes) {
            Ok(audio) if audio.len() >= from + new * HOP => speech.extend_from_slice(&audio[from..from + new * HOP]),
            _ => speech.extend(std::iter::repeat_n(0.0, new * HOP)),
        }
        let keep = codes.len().min(CONTEXT);
        self.decoded_before = codes[codes.len() - keep..].to_vec();
    }
}

/// A frame's codebook indices, ten bits each, most significant first.
fn pack(code: &[u32]) -> Vec<u8> {
    code.iter().flat_map(|&c| (0..INDEX_BITS).rev().map(move |i| ((c >> i) & 1) as u8)).collect()
}

fn unpack(bits: &[u8], codebooks: usize) -> Vec<u32> {
    (0..codebooks)
        .map(|b| bits[b * INDEX_BITS..(b + 1) * INDEX_BITS].iter().fold(0u32, |acc, &x| (acc << 1) | u32::from(x & 1)))
        .collect()
}

impl SpeechCodec for EncodecSpeech {
    fn codec(&self) -> Codec {
        self.codec
    }

    fn sample_rate(&self) -> u32 {
        24_000
    }

    fn encode(&mut self, speech: &[f32], frames: &mut Vec<Vec<u8>>) {
        self.input.extend_from_slice(speech);
        while self.input.len() >= CHUNK * HOP {
            let chunk: Vec<f32> = self.input.drain(..CHUNK * HOP).collect();
            self.encode_chunk(&chunk, frames);
        }
    }

    fn finish_encoding(&mut self, frames: &mut Vec<Vec<u8>>) {
        if self.input.is_empty() {
            return;
        }
        // The speaker's last words, made up to whole frames with silence.
        let mut chunk = std::mem::take(&mut self.input);
        chunk.resize(chunk.len().div_ceil(HOP) * HOP, 0.0);
        self.encode_chunk(&chunk, frames);
        self.heard_before.clear();
    }

    fn decode(&mut self, frame: Option<&[u8]>, speech: &mut Vec<f32>) {
        match frame {
            Some(bits) => {
                self.pending.push(unpack(bits, self.codebooks));
                if self.pending.len() >= CHUNK {
                    self.decode_pending(speech);
                }
            }
            None => {
                // What came before a gap is played; what comes after starts
                // afresh, with nothing in front of it that is not there.
                self.decode_pending(speech);
                self.decoded_before.clear();
                speech.extend(std::iter::repeat_n(0.0, HOP));
            }
        }
    }

    fn finish_decoding(&mut self, speech: &mut Vec<f32>) {
        self.decode_pending(speech);
        self.decoded_before.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A stand-in network: each frame's code is its mean level, quantised;
    /// decoding makes a frame of that level. Enough to see that chunking,
    /// context and packing put every frame back where it came from.
    #[derive(Debug)]
    struct Levels;

    impl Network for Levels {
        fn encode(&self, audio: &[f32], codebooks: usize) -> Result<Vec<Vec<u32>>, String> {
            Ok(audio
                .chunks(HOP)
                .map(|f| {
                    let mean = f.iter().sum::<f32>() / f.len() as f32;
                    vec![((mean + 1.0) * 500.0).round().clamp(0.0, 1023.0) as u32; codebooks]
                })
                .collect())
        }

        fn decode(&self, codes: &[Vec<u32>]) -> Result<Vec<f32>, String> {
            Ok(codes.iter().flat_map(|c| std::iter::repeat_n(c[0] as f32 / 500.0 - 1.0, HOP)).collect())
        }
    }

    fn codec() -> EncodecSpeech {
        EncodecSpeech::with_network(Codec::Encodec1500, Arc::new(Levels)).unwrap()
    }

    #[test]
    fn indices_pack_and_unpack() {
        let code = vec![0, 1023, 517, 42];
        assert_eq!(unpack(&pack(&code), 4), code);
        assert_eq!(pack(&code).len(), 40);
    }

    #[test]
    fn frames_come_out_in_chunks_and_go_back_in_order() {
        // A staircase: each frame a level of its own.
        let frames_in = 100;
        let speech: Vec<f32> = (0..frames_in).flat_map(|f| std::iter::repeat_n(f as f32 / 200.0, HOP)).collect();
        let mut tx = codec();
        let mut frames = Vec::new();
        // Fed in awkward pieces, as a sound card would.
        for piece in speech.chunks(777) {
            tx.encode(piece, &mut frames);
        }
        assert_eq!(frames.len(), frames_in / CHUNK * CHUNK);
        tx.finish_encoding(&mut frames);
        assert_eq!(frames.len(), frames_in);
        assert!(frames.iter().all(|f| f.len() == Codec::Encodec1500.bits()));

        let mut rx = codec();
        let mut back = Vec::new();
        for f in &frames {
            rx.decode(Some(f), &mut back);
        }
        rx.finish_decoding(&mut back);
        assert_eq!(back.len(), speech.len());
        for (i, (a, b)) in speech.iter().zip(&back).enumerate().step_by(HOP) {
            assert!((a - b).abs() < 0.002, "frame {} came back at {b}, not {a}", i / HOP);
        }
    }

    /// Weights of the right shapes and nothing else, for timing the model
    /// without Meta's file.
    struct Random;

    impl candle_nn::var_builder::SimpleBackend for Random {
        fn get(
            &self,
            s: candle_core::Shape,
            name: &str,
            _h: candle_nn::Init,
            dtype: DType,
            dev: &Device,
        ) -> candle_core::Result<Tensor> {
            if name.ends_with("weight_g") {
                Tensor::ones(s, dtype, dev)
            } else {
                Tensor::randn(0f32, 0.05, s, dev)?.to_dtype(dtype)
            }
        }

        fn contains_tensor(&self, _name: &str) -> bool {
            true
        }
    }

    /// How fast the real model runs a chunk each way on this machine. The
    /// weights are random, which costs the same as Meta's.
    #[test]
    #[ignore = "timing, not correctness: cargo test --release -p voice -- --ignored --nocapture"]
    fn the_model_keeps_up_with_speech() {
        let vb = VarBuilder::from_backend(Box::new(Random), DType::F32, Device::Cpu);
        let model = Model::new(&Config::default(), vb).expect("the model builds");
        let network: Arc<dyn Network> = Arc::new(Encodec { model });
        let mut tx = EncodecSpeech::with_network(Codec::Encodec1500, network.clone()).unwrap();
        let mut rx = EncodecSpeech::with_network(Codec::Encodec1500, network).unwrap();
        let speech = crate::synthetic_speech(4.0, 24_000.0);
        let started = std::time::Instant::now();
        let mut frames = Vec::new();
        tx.encode(&speech, &mut frames);
        let encoding = started.elapsed().as_secs_f64();
        let started = std::time::Instant::now();
        let mut back = Vec::new();
        for f in &frames {
            rx.decode(Some(f), &mut back);
        }
        let decoding = started.elapsed().as_secs_f64();
        println!("4 s of speech: encoded in {encoding:.2} s, decoded in {decoding:.2} s, {} frames", frames.len());
        assert!(encoding < 4.0 && decoding < 4.0, "slower than real time");
    }

    /// The lag, in samples, at which `b` best matches `a`, within `most`
    /// either way, and the correlation there.
    fn best_lag(a: &[f32], b: &[f32], most: i64) -> (i64, f64) {
        (-most..=most)
            .map(|lag| {
                let (mut ab, mut aa, mut bb) = (0.0f64, 0.0f64, 0.0f64);
                for (i, &x) in a.iter().enumerate() {
                    let Some(&y) = usize::try_from(i as i64 + lag).ok().and_then(|j| b.get(j)) else { continue };
                    ab += f64::from(x) * f64::from(y);
                    aa += f64::from(x) * f64::from(x);
                    bb += f64::from(y) * f64::from(y);
                }
                (lag, ab / (aa * bb).sqrt().max(1e-12))
            })
            .max_by(|p, q| p.1.total_cmp(&q.1))
            .unwrap_or((0, 0.0))
    }

    /// With Meta's weights, if they are here: the chunked stream against the
    /// model run on the whole clip at once, which is what the stream is
    /// standing in for. `VOICEMODEM_SPEECH` may name a WAV of real speech.
    #[test]
    #[ignore = "needs Meta's weights: cargo test --release -p voice -- --ignored --nocapture"]
    fn chunks_match_the_whole_clip() {
        let Ok(network) = shared() else {
            println!("no EnCodec weights here; nothing to check");
            return;
        };
        let speech: Vec<f32> = match std::env::var("VOICEMODEM_SPEECH") {
            Ok(path) => {
                let wav = line::wav::read(&path).expect("the speech WAV reads");
                let mut out = Vec::new();
                crate::Rate::new(f64::from(wav.sample_rate), 24_000.0).process(&wav.mono(), &mut out);
                out
            }
            Err(_) => crate::synthetic_speech(6.0, 24_000.0),
        };
        let speech = &speech[..speech.len() / HOP * HOP];
        for codec in [Codec::Encodec1500, Codec::Encodec3000] {
            let books = codec.bits() / INDEX_BITS;
            let whole = network.decode(&network.encode(speech, books).unwrap()).unwrap();
            let mut tx = EncodecSpeech::with_network(codec, network.clone()).unwrap();
            let mut rx = EncodecSpeech::with_network(codec, network.clone()).unwrap();
            let (mut frames, mut chunked) = (Vec::new(), Vec::new());
            tx.encode(speech, &mut frames);
            tx.finish_encoding(&mut frames);
            for f in &frames {
                rx.decode(Some(f), &mut chunked);
            }
            rx.finish_decoding(&mut chunked);
            let (lag_whole, r_whole) = best_lag(speech, &whole, 2000);
            let (lag_chunked, r_chunked) = best_lag(speech, &chunked, 2000);
            let (lag_between, r_between) = best_lag(&whole, &chunked, 2000);
            println!(
                "{}: whole clip {lag_whole:+} samples, r {r_whole:.3}; chunked {lag_chunked:+}, r {r_chunked:.3}; \
                 chunked against whole {lag_between:+}, r {r_between:.3}; {} and {} samples for {}",
                codec.label(),
                whole.len(),
                chunked.len(),
                speech.len()
            );
            assert_eq!(lag_between, 0, "the chunked stream is shifted against the whole clip");
        }
    }

    #[test]
    fn a_lost_frame_is_a_frame_of_silence() {
        let mut rx = codec();
        let mut back = Vec::new();
        rx.decode(Some(&pack(&[750, 750])), &mut back);
        rx.decode(None, &mut back);
        assert_eq!(back.len(), 2 * HOP);
        assert!(back[..HOP].iter().all(|x| (x - 0.5).abs() < 1e-6));
        assert!(back[HOP..].iter().all(|x| *x == 0.0));
    }
}
