//! voicemodem: digital voice over a radio's voice channel -- SSB or FM,
//! satellite or terrestrial -- in BPSK, QPSK and 8PSK, each burst led by a
//! DBPSK preamble that carries everything the receiver needs.
//!
//! The pieces, in the order speech meets them:
//!
//! - [`voice`]: which codec in which mode, codec frames packed into
//!   codewords, and a transmission paced as it is spoken. ([`transfer`] does
//!   the same for a text message or a file, cut into numbered blocks.)
//! - [`frame`]: the preamble and its header, the pilots, and the codewords
//!   -- whitening, K=7 convolutional code at 1/2, 2/3 or 3/4, interleaving,
//!   Gray-mapped PSK ([`fec`], [`psk`]).
//! - [`tx`]: root-raised-cosine pulses at 1600 baud on 1500 Hz for SSB, or
//!   2400 baud on 1800 Hz for FM ([`profile`] has every fixed number).
//! - [`detect`]: the preamble found differentially -- no carrier needed --
//!   and its header read.
//! - [`rx`]: a fresh copy of BinModem's QAM core for each burst, trained by
//!   least squares on the preamble's 300 known symbols and tracking from
//!   there; [`framer`] finds the pilots, catches slips and erases what they
//!   spoil; the codewords are decoded and the blocks put back together.
//! - [`channel`]: simulated lines for the tests and the self test.
//!
//! Streaming throughout, as BinModem's rule is: the receiver takes one
//! sample at a time and never re-acquires at a buffer boundary.

pub mod channel;
pub mod detect;
pub mod fec;
pub mod frame;
pub mod framer;
pub mod profile;
pub mod psk;
pub mod rx;
pub mod transfer;
pub mod tx;
pub mod voice;

pub use fec::Rate;
pub use frame::{Geometry, Header, Kind};
pub use profile::Profile;
pub use psk::Modulation;
pub use rx::{BurstReport, Event, Receiver, State};
pub use transfer::{Assembler, Content, Delivery, Outgoing};
pub use tx::Modulator;
pub use voice::{Codec, VOICE_MODES, VoiceCodeword, VoiceMode, VoiceTx};
