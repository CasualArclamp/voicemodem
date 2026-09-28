//! Streaming DSP primitives shared by every modulation.
//!
//! Copied from BinModem (github.com/CasualArclamp/BinModem, `crates/dsp` at
//! d9bb18e) so that voicemodem's receiver is the same live-proven code. Two
//! additions: [`qam::Core::half_near`], because voicemodem finds its own
//! preamble and tells training where it starts; and
//! [`qam::Options::least_trained_db`], whose default is BinModem's fixed
//! 6 dB, because voicemodem's robust modes train below it. Keep it otherwise
//! identical, so that fixes can move between the two by diff.
//!
//! Design rule for this crate: everything is sample-at-a-time and stateful.
//! A modem's timing recovery, carrier tracking, equaliser and echo canceller
//! are continuous adaptive loops that must never be reset at a buffer
//! boundary, so no public API here takes or returns a block of samples.
//!
//! This is the single most important departure from the previous attempt,
//! whose `modulate(bits) -> samples` / `demodulate(samples) -> bits` shape
//! forced every loop to re-acquire on each block.

pub mod complex;
pub mod echo;
pub mod equalizer;
pub mod fft;
pub mod filter;
pub mod fsk;
pub mod nco;
pub mod qam;
pub mod resample;
pub mod shaping;
pub mod tone;

pub use complex::{Complex, least_squares, solve_hermitian};
pub use echo::{EchoCanceller, EchoFinder, Reflection};
pub use equalizer::Equalizer;
pub use fft::{Fft, Spectrum};
pub use filter::{Biquad, Cascade, OnePole, bandpass, butter_highpass, butter_lowpass};
pub use fsk::FskDetector;
pub use nco::Nco;
pub use resample::Resampler;
pub use shaping::{
    ComplexFir, Fir, Gardner, fir_lowpass, fir_lowpass_kaiser, rrc_at, rrc_taps,
};
pub use tone::{ReversalDetector, ToneDetector};
