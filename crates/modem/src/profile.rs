//! The waveform's fixed numbers: everything both ends must agree on without
//! being told.
//!
//! Two profiles, for the two kinds of radio voice channel:
//!
//! - **Narrow**, for SSB -- HF, and linear-transponder satellites: 1600 baud
//!   on 1500 Hz with a 20% roll-off fills 540 to 2460 Hz, inside a 2.4 kHz
//!   filter's 300 to 2700 with room at each edge. SSB passes a mistuning or
//!   a satellite's residual Doppler straight through to the audio, so the
//!   detector looks at three carrier offsets at once and the receiver
//!   follows what is left.
//! - **Wide**, for FM -- repeaters, FM satellites, a rig's flat data port:
//!   2400 baud on 1800 Hz, 360 to 3240 Hz. An FM discriminator takes the
//!   carrier's offset out before the audio, so there is none to follow.
//!
//! Both run at 16 kHz, as BinModem does: its QAM core's interpolating filter
//! leaves the mixer image at least 39 dB down there, a sound card's 48 kHz
//! divides into it exactly, and 1600 baud is exactly ten samples a symbol.
//! The receiver listens for both profiles at once; the preamble it finds
//! says which one it was.

/// Line samples a second.
pub const FS: f64 = 16_000.0;
/// Root-raised-cosine roll-off, at both ends, both profiles.
pub const ROLLOFF: f64 = 0.2;

/// A radio channel's width, and the symbol rate and carrier that fill it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Profile {
    /// SSB: 1600 baud on 1500 Hz.
    Narrow,
    /// FM: 2400 baud on 1800 Hz.
    Wide,
}

impl Profile {
    pub const ALL: [Profile; 2] = [Profile::Narrow, Profile::Wide];

    /// Symbols a second.
    pub fn baud(self) -> f64 {
        match self {
            Profile::Narrow => 1600.0,
            Profile::Wide => 2400.0,
        }
    }

    /// The carrier, in hertz.
    pub fn carrier(self) -> f64 {
        match self {
            Profile::Narrow => 1500.0,
            Profile::Wide => 1800.0,
        }
    }

    /// Line samples a symbol: ten for narrow, 6 2/3 for wide.
    pub fn sps(self) -> f64 {
        FS / self.baud()
    }

    /// The occupied band, in hertz.
    pub fn band(self) -> (f64, f64) {
        let half = self.baud() * (1.0 + ROLLOFF) / 2.0;
        (self.carrier() - half, self.carrier() + half)
    }

    /// Carrier offsets the detector looks at, each good for about a quarter
    /// of the symbol rate either side: narrow covers about +-450 Hz of
    /// mistuning or Doppler, wide expects none.
    pub fn search_offsets(self) -> &'static [f64] {
        match self {
            Profile::Narrow => &[-300.0, 0.0, 300.0],
            Profile::Wide => &[0.0],
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Profile::Narrow => "Narrow (SSB)",
            Profile::Wide => "Wide (FM)",
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Profile::Narrow => "narrow",
            Profile::Wide => "wide",
        }
    }

    pub fn parse(text: &str) -> Option<Self> {
        match text.trim().to_ascii_lowercase().as_str() {
            "narrow" | "ssb" | "n" => Some(Profile::Narrow),
            "wide" | "fm" | "w" => Some(Profile::Wide),
            _ => None,
        }
    }

    /// Seconds of `symbols`.
    pub fn seconds(self, symbols: usize) -> f64 {
        symbols as f64 / self.baud()
    }

    /// Seconds a burst of `codewords` takes on the line.
    pub fn burst_seconds(self, codewords: usize) -> f64 {
        self.seconds(burst_symbols(codewords))
    }
}

/// Symbols of reversals at the head of the preamble. A radio's AGC, an FM
/// squelch, a VOX and a far receiver's noise blanker all get these to settle
/// on; nothing needs all of them to arrive.
pub const DOTS: usize = 64;
/// Symbols of the unique word: a 63-chip m-sequence, differentially encoded.
pub const UW: usize = 63;
/// The header, before coding: eight bytes and a CRC-16.
pub const HEADER_BYTES: usize = 8;
pub const HEADER_BITS: usize = 8 * (HEADER_BYTES + 2);
/// The header after rate-1/2 coding with its tail: one DBPSK symbol a bit.
pub const HEADER_SYMBOLS: usize = 2 * (HEADER_BITS + crate::fec::TAIL);
/// Preamble symbol the unique word ends on: after the reference symbol and
/// the reversals.
pub const UW_LAST: usize = DOTS + UW;
/// The whole preamble: a reference symbol, reversals, unique word, header.
pub const PREAMBLE: usize = 1 + DOTS + UW + HEADER_SYMBOLS;

/// Pilot symbols at the head of every slot, and the slot's length: known
/// symbols every 80 ms narrow and 53 ms wide, for the constellation's phase
/// after a fade and the symbols' place after a slip, at a cost of an eighth.
pub const PILOT: usize = 16;
pub const SLOT: usize = 128;
pub const DATA: usize = SLOT - PILOT;
/// Slots a codeword's data spans, and its data symbols: 640 ms narrow,
/// 427 ms wide, interleaved end to end against a fade.
pub const SLOTS_PER_CODEWORD: usize = 8;
pub const CODEWORD_SYMBOLS: usize = DATA * SLOTS_PER_CODEWORD;
/// The most codewords one preamble announces. A longer transmission is sent
/// as several bursts back to back, so that a receiver that missed one
/// preamble, or lost one burst, is back within seconds.
pub const MAX_CODEWORDS: usize = 32;

/// Symbols in a burst of `codewords`: preamble, slots, and the closing
/// pilot that confirms the last slot's alignment.
pub fn burst_symbols(codewords: usize) -> usize {
    PREAMBLE + codewords * SLOTS_PER_CODEWORD * SLOT + PILOT
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn narrow_fits_an_ssb_filter_and_wide_a_telephone_channel() {
        let (low, high) = Profile::Narrow.band();
        assert!(low >= 500.0 && high <= 2500.0, "narrow fills {low}..{high}");
        let (low, high) = Profile::Wide.band();
        assert!(low >= 300.0 && high <= 3400.0, "wide fills {low}..{high}");
        assert_eq!(Profile::Narrow.sps(), 10.0);
    }
}
