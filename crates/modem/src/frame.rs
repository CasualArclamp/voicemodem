//! What goes on the line, symbol by symbol: the DBPSK preamble and its
//! header, the pilots, and the codewords between them.
//!
//! ```text
//!  preamble, DBPSK, 300 symbols                     payload, coherent PSK
//! +---+-----------+-----------+--------------+     +-------+---------+-------+-----+-------+
//! | R | reversals | unique    | header       |     | pilot | data    | pilot | ... | pilot |
//! |   | 64        | word 63   | 172 (coded)  |     | 16    | 112     | 16    |     | 16    |
//! +---+-----------+-----------+--------------+     +-------+---------+-------+-----+-------+
//!                                                   '-------- slot, 128 --------'
//! ```
//!
//! Everything in the preamble is differential: each symbol is the one before
//! it, reversed or not. So all of it can be read before the carrier's phase
//! or frequency is known -- the product of each symbol with the one before
//! carries the bit, and the carrier's offset only turns every product by the
//! same angle. The reversals give a receiver's gain and timing something to
//! settle on; the unique word, correlated against those products, says where
//! the symbols are to a fraction of one and how far the carrier is off; and
//! the header says what follows. Once the header's CRC passes, every symbol
//! of the preamble is known, and the receiver trains its equaliser on all
//! 300 of them by least squares.
//!
//! The payload is coherent. Every slot starts with sixteen pilots, a long
//! pseudo-random sequence so no two nearby slots share them, sent on a point
//! of the payload's constellation and its opposite: +-1, or +-(1 + j)/sqrt 2
//! for QPSK, whose points are on the diagonals. The receiver finds each
//! slot's pilots, which says whether the symbols have slipped (a jitter
//! buffer adding or dropping 20 ms moves everything after it by 48 symbols)
//! and which way round the constellation is.

use std::sync::OnceLock;

use dsp::Complex;

use crate::fec::{self, Interleaver, Rate};
use crate::profile::{
    DATA, DOTS, HEADER_BITS, HEADER_BYTES, HEADER_SYMBOLS, MAX_CODEWORDS, PILOT, PREAMBLE, Profile,
    SLOTS_PER_CODEWORD, UW,
};
use crate::psk::Modulation;

/// What the codewords of a burst carry.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Kind {
    /// Numbered blocks of a text message or a file.
    Data,
    /// Speech codec frames, as they are spoken.
    Voice,
}

/// What a preamble announces.
///
/// ```text
/// bit  0      version, nought
///      1      kind: data or voice
///      2-3    modulation
///      4-5    code rate
///      6-9    codec (voice), nought (data)
///      10-15  codewords in this burst
///      16-31  stream: which transfer, or which transmission
///      32-47  sequence: the first block (data), the burst's number (voice)
///      48-63  total: the transfer's blocks (data), flags (voice)
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Header {
    pub kind: Kind,
    pub modulation: Modulation,
    pub rate: Rate,
    /// Which speech codec a voice burst's frames are; nought for data.
    pub codec: u8,
    /// Codewords in this burst, 1 to [`MAX_CODEWORDS`]. A voice burst may
    /// stop short of it when the transmitter is unkeyed.
    pub codewords: u8,
    /// Which transfer or transmission the burst belongs to.
    pub stream: u16,
    /// Data: the block number of the burst's first codeword. Voice: the
    /// burst's number within its transmission.
    pub sequence: u16,
    /// Data: how many blocks the whole transfer has. Voice: flags, nought.
    pub total: u16,
}

const VERSION: u64 = 0;

impl Header {
    /// The eight header bytes and their CRC-16.
    pub fn to_bytes(self) -> [u8; HEADER_BYTES + 2] {
        let kind = match self.kind {
            Kind::Data => 0,
            Kind::Voice => 1,
        };
        let word = VERSION
            | (kind << 1)
            | (u64::from(self.modulation.code()) << 2)
            | (u64::from(self.rate.code()) << 4)
            | (u64::from(self.codec & 0x0F) << 6)
            | (u64::from(self.codewords & 0x3F) << 10)
            | (u64::from(self.stream) << 16)
            | (u64::from(self.sequence) << 32)
            | (u64::from(self.total) << 48);
        let mut out = [0u8; HEADER_BYTES + 2];
        out[..HEADER_BYTES].copy_from_slice(&word.to_le_bytes());
        let crc = fec::crc16(&out[..HEADER_BYTES]);
        out[HEADER_BYTES..].copy_from_slice(&crc.to_le_bytes());
        out
    }

    /// A header from its bytes, if the CRC holds and every field makes sense.
    pub fn from_bytes(bytes: &[u8]) -> Option<Self> {
        if bytes.len() < HEADER_BYTES + 2 {
            return None;
        }
        let crc = u16::from_le_bytes([bytes[HEADER_BYTES], bytes[HEADER_BYTES + 1]]);
        if fec::crc16(&bytes[..HEADER_BYTES]) != crc {
            return None;
        }
        let word = u64::from_le_bytes(bytes[..HEADER_BYTES].try_into().ok()?);
        if word & 1 != VERSION {
            return None;
        }
        let header = Self {
            kind: if (word >> 1) & 1 == 0 { Kind::Data } else { Kind::Voice },
            modulation: Modulation::from_code(((word >> 2) & 3) as u8)?,
            rate: Rate::from_code(((word >> 4) & 3) as u8)?,
            codec: ((word >> 6) & 0x0F) as u8,
            codewords: ((word >> 10) & 0x3F) as u8,
            stream: (word >> 16) as u16,
            sequence: (word >> 32) as u16,
            total: (word >> 48) as u16,
        };
        let count = (1..=MAX_CODEWORDS).contains(&usize::from(header.codewords));
        let sane = match header.kind {
            Kind::Data => {
                header.codec == 0
                    && u32::from(header.sequence) + u32::from(header.codewords) <= u32::from(header.total)
            }
            Kind::Voice => true,
        };
        (count && sane).then_some(header)
    }

    /// The header's code bits in the order they go out: coded at rate 1/2
    /// with its tail, and interleaved.
    pub fn code_bits(self) -> Vec<u8> {
        let mut bits = fec::bits_of(&self.to_bytes());
        bits.extend([0; fec::TAIL]);
        let code = fec::encode(&bits, Rate::Half);
        debug_assert_eq!(code.len(), HEADER_SYMBOLS);
        header_interleaver().interleave(&code)
    }

    /// A header from soft values of its code bits in the order they came.
    pub fn decode(soft: &[f64]) -> Option<Self> {
        if soft.len() != HEADER_SYMBOLS {
            return None;
        }
        let code = header_interleaver().deinterleave(soft);
        let bits = fec::decode(&code, HEADER_BITS + fec::TAIL, Rate::Half);
        Self::from_bytes(&fec::bytes_of(&bits[..HEADER_BITS]))
    }

    pub fn geometry(self) -> Geometry {
        Geometry::new(self.modulation, self.rate)
    }
}

fn header_interleaver() -> &'static Interleaver {
    static IL: OnceLock<Interleaver> = OnceLock::new();
    IL.get_or_init(|| Interleaver::new(HEADER_SYMBOLS))
}

/// The unique word's 63 chips: the m-sequence of x^6 + x + 1.
pub fn unique_word() -> &'static [u8; UW] {
    static WORD: OnceLock<[u8; UW]> = OnceLock::new();
    WORD.get_or_init(|| {
        let mut state: u8 = 1;
        let mut word = [0u8; UW];
        for chip in &mut word {
            let out = state & 1;
            *chip = out;
            // x^6 + x + 1, as a Fibonacci register shifting right.
            let feedback = (state ^ (state >> 1)) & 1;
            state = (state >> 1) | (feedback << 5);
        }
        word
    })
}

/// The preamble's differential signs: for each symbol after the reference,
/// +1 if it repeats the one before and -1 if it reverses it.
pub fn preamble_signs(header: Header) -> Vec<f64> {
    let mut signs = Vec::with_capacity(PREAMBLE - 1);
    signs.extend(std::iter::repeat_n(-1.0, DOTS));
    signs.extend(unique_word().iter().map(|&c| if c == 1 { -1.0 } else { 1.0 }));
    signs.extend(header.code_bits().iter().map(|&c| if c == 1 { -1.0 } else { 1.0 }));
    signs
}

/// The unique word's differential signs alone, as the detector correlates
/// them.
pub fn unique_word_signs() -> Vec<f64> {
    unique_word().iter().map(|&c| if c == 1 { -1.0 } else { 1.0 }).collect()
}

/// The preamble's symbols: +-1, differentially encoded from +1.
pub fn preamble(header: Header) -> Vec<Complex> {
    let mut symbols = Vec::with_capacity(PREAMBLE);
    let mut last = 1.0;
    symbols.push(Complex::new(last, 0.0));
    for sign in preamble_signs(header) {
        last *= sign;
        symbols.push(Complex::new(last, 0.0));
    }
    debug_assert_eq!(symbols.len(), PREAMBLE);
    symbols
}

/// The pilots of slot `slot` as chips of +-1: sixteen of the m-sequence of
/// x^11 + x^9 + 1, taken in turn, so that no two slots of a burst have the
/// same pilots and a slip of a whole slot cannot pass for none. They go out
/// as [`pilot_symbols`].
pub fn pilots(slot: usize) -> [f64; PILOT] {
    static SEQUENCE: OnceLock<Vec<f64>> = OnceLock::new();
    let sequence = SEQUENCE.get_or_init(|| {
        let mut state: u16 = 0x2A5;
        (0..2047)
            .map(|_| {
                let out = state & 1;
                let feedback = (state ^ (state >> 2)) & 1;
                state = (state >> 1) | (feedback << 10);
                if out == 1 { -1.0 } else { 1.0 }
            })
            .collect()
    });
    let mut out = [0.0; PILOT];
    for (j, slot_chip) in out.iter_mut().enumerate() {
        *slot_chip = sequence[(slot * PILOT + j) % sequence.len()];
    }
    out
}

/// How a codeword is laid out for one modulation and rate.
///
/// A codeword fills the data symbols of a whole number of slots: eight for a
/// data transfer ([`crate::profile::CODEWORD_SYMBOLS`]), whose blocks want the interleaving
/// and do not mind the wait, and fewer for speech, which minds the wait -- a
/// codeword can only be sent once its speech has been spoken and only played
/// once it has all arrived. What changes with the mode is how much fits. A
/// data codeword's bytes are a block number, the block's payload and a
/// CRC-32.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Geometry {
    pub modulation: Modulation,
    pub rate: Rate,
    /// Slots a codeword spans, and the interleaver across its channel bits.
    pub slots: usize,
    interleaver: std::sync::Arc<Interleaver>,
    /// Bits the data symbols carry.
    pub channel_bits: usize,
    /// Encoder input bits, tail included: a whole number of puncturing
    /// periods.
    pub inputs: usize,
    /// Code bits sent; the few channel bits after them are filler.
    pub coded: usize,
    /// Whole bytes before the tail: block number, payload, CRC.
    pub bytes: usize,
    /// Payload bytes a codeword carries.
    pub payload: usize,
}

/// Bytes of block number and CRC around each payload.
const BLOCK_OVERHEAD: usize = 2 + 4;

impl Geometry {
    /// A data transfer's codeword: [`SLOTS_PER_CODEWORD`] slots.
    pub fn new(modulation: Modulation, rate: Rate) -> Self {
        Self::with_slots(modulation, rate, SLOTS_PER_CODEWORD)
    }

    /// A codeword spanning `slots` slots.
    pub fn with_slots(modulation: Modulation, rate: Rate, slots: usize) -> Self {
        let channel_bits = slots * DATA * modulation.bits();
        let (per, sent) = rate.period();
        let periods = channel_bits / sent;
        let inputs = periods * per;
        let coded = periods * sent;
        let bytes = (inputs - fec::TAIL) / 8;
        Self {
            modulation,
            rate,
            slots,
            interleaver: std::sync::Arc::new(Interleaver::new(channel_bits)),
            channel_bits,
            inputs,
            coded,
            bytes,
            payload: bytes.saturating_sub(BLOCK_OVERHEAD),
        }
    }

    /// Data symbols a codeword carries.
    pub fn data_symbols(&self) -> usize {
        self.slots * DATA
    }

    /// Payload bits a second while the payload is going out, pilots
    /// included but not the preamble.
    pub fn bit_rate(&self, profile: Profile) -> f64 {
        8.0 * self.payload as f64 / self.seconds(profile)
    }

    /// Seconds a codeword takes on the line, its pilots included.
    pub fn seconds(&self, profile: Profile) -> f64 {
        profile.seconds(self.slots * crate::profile::SLOT)
    }

    /// Bits a codeword carries before its tail.
    pub fn info_bits(&self) -> usize {
        self.inputs - fec::TAIL
    }

    fn interleaver(&self) -> &Interleaver {
        &self.interleaver
    }

    /// The data symbols of a codeword carrying `bits`, which are padded with
    /// noughts to [`Geometry::info_bits`], whitened, coded and interleaved.
    pub fn encode_bits(&self, bits: &[u8]) -> Vec<Complex> {
        assert!(bits.len() <= self.info_bits(), "{} bits into a {}-bit codeword", bits.len(), self.info_bits());
        let mut bits = bits.to_vec();
        bits.resize(self.info_bits(), 0);
        fec::whiten_bits(&mut bits);
        bits.resize(self.inputs, 0);
        let mut code = fec::encode(&bits, self.rate);
        code.resize(self.channel_bits, 0);
        let channel = self.interleaver().interleave(&code);
        let m = self.modulation.bits();
        channel
            .chunks(m)
            .map(|chunk| self.modulation.map(chunk.iter().fold(0usize, |acc, &b| (acc << 1) | usize::from(b))))
            .collect()
    }

    /// A codeword's [`Geometry::info_bits`] bits from the soft values of its
    /// channel bits, in the order they came off the line. Nothing checks
    /// them; whatever they carry has its own CRC.
    pub fn decode_bits(&self, soft: &[f64]) -> Option<Vec<u8>> {
        if soft.len() != self.channel_bits {
            return None;
        }
        let code = self.interleaver().deinterleave(soft);
        let mut bits = fec::decode(&code[..self.coded], self.inputs, self.rate);
        bits.truncate(self.info_bits());
        fec::whiten_bits(&mut bits);
        Some(bits)
    }

    /// The data symbols of block `index` carrying `payload`, which is padded
    /// with noughts to [`Geometry::payload`] bytes.
    pub fn encode(&self, index: u16, payload: &[u8]) -> Vec<Complex> {
        assert!(payload.len() <= self.payload, "{} bytes into a {}-byte block", payload.len(), self.payload);
        let mut bytes = Vec::with_capacity(self.bytes);
        bytes.extend_from_slice(&index.to_le_bytes());
        bytes.extend_from_slice(payload);
        bytes.resize(2 + self.payload, 0);
        let crc = fec::crc32(&bytes);
        bytes.extend_from_slice(&crc.to_le_bytes());
        self.encode_bits(&fec::bits_of(&bytes))
    }

    /// A block from the soft values of its channel bits, in the order they
    /// came off the line: its number and payload if the CRC holds.
    pub fn decode(&self, soft: &[f64]) -> Option<(u16, Vec<u8>)> {
        let bits = self.decode_bits(soft)?;
        let bytes = fec::bytes_of(&bits[..8 * self.bytes]);
        let (body, crc) = bytes.split_at(self.bytes - 4);
        let crc = u32::from_le_bytes(crc.try_into().ok()?);
        if fec::crc32(body) != crc {
            return None;
        }
        let index = u16::from_le_bytes([body[0], body[1]]);
        Some((index, body[2..].to_vec()))
    }
}

/// The pilots of slot `slot` as symbols, on `modulation`'s pilot point and
/// its opposite.
pub fn pilot_symbols(modulation: Modulation, slot: usize) -> impl Iterator<Item = Complex> {
    let point = modulation.pilot();
    pilots(slot).into_iter().map(move |p| point.scale(p))
}

/// A codeword's symbols as they go out: each of its slots' pilots and then
/// its share of `data`, a whole number of slots' worth of `modulation`, the
/// first slot numbered `first_slot` in the burst.
pub fn codeword_slots(modulation: Modulation, first_slot: usize, data: &[Complex]) -> Vec<Complex> {
    assert!(!data.is_empty() && data.len().is_multiple_of(DATA), "{} data symbols is not whole slots", data.len());
    let mut symbols = Vec::with_capacity(data.len() / DATA * (PILOT + DATA));
    for (i, chunk) in data.chunks(DATA).enumerate() {
        symbols.extend(pilot_symbols(modulation, first_slot + i));
        symbols.extend_from_slice(chunk);
    }
    symbols
}

/// Every symbol of a burst: the preamble, then for each slot its pilots and
/// its share of the codewords' data, then the closing pilots.
pub fn burst(header: Header, codewords: &[Vec<Complex>]) -> Vec<Complex> {
    assert_eq!(codewords.len(), usize::from(header.codewords), "the header announces a different count");
    let mut symbols = preamble(header);
    let mut slot = 0;
    for codeword in codewords {
        symbols.extend(codeword_slots(header.modulation, slot, codeword));
        slot += codeword.len() / DATA;
    }
    symbols.extend(pilot_symbols(header.modulation, slot));
    symbols
}

#[cfg(test)]
mod tests {
    use super::*;

    fn header() -> Header {
        Header {
            kind: Kind::Data,
            modulation: Modulation::Psk8,
            rate: Rate::ThreeQuarters,
            codec: 0,
            codewords: 5,
            stream: 0xBEEF,
            sequence: 1234,
            total: 5000,
        }
    }

    #[test]
    fn a_voice_header_round_trips_too() {
        let h = Header { kind: Kind::Voice, codec: 5, sequence: 9, total: 1, ..header() };
        assert_eq!(Header::from_bytes(&h.to_bytes()), Some(h));
    }

    #[test]
    fn bits_round_trip_through_a_codeword() {
        let g = Geometry::new(Modulation::Qpsk, Rate::TwoThirds);
        let bits: Vec<u8> = (0..g.info_bits()).map(|i| ((i * 5 + i / 7) % 3 == 0) as u8).collect();
        let mut soft = Vec::new();
        for z in g.encode_bits(&bits) {
            Modulation::Qpsk.demap(z, 0.5, &mut soft);
        }
        assert_eq!(g.decode_bits(&soft), Some(bits));
    }

    #[test]
    fn header_round_trips_through_its_code() {
        let h = header();
        let soft: Vec<f64> = h.code_bits().iter().map(|&c| if c == 0 { 1.0 } else { -1.0 }).collect();
        assert_eq!(Header::decode(&soft), Some(h));
    }

    #[test]
    fn a_damaged_header_is_refused_not_misread() {
        let h = header();
        let mut bytes = h.to_bytes();
        bytes[3] ^= 0x10;
        assert_eq!(Header::from_bytes(&bytes), None);
    }

    #[test]
    fn the_unique_word_is_an_m_sequence() {
        let w = unique_word();
        let ones = w.iter().filter(|c| **c == 1).count();
        assert_eq!(ones, 32);
        // Two-valued periodic autocorrelation: 63 at nought, -1 elsewhere.
        let s = unique_word_signs();
        for shift in 1..UW {
            let r: f64 = (0..UW).map(|i| s[i] * s[(i + shift) % UW]).sum();
            assert_eq!(r, -1.0, "shift {shift}");
        }
    }

    #[test]
    fn neighbouring_slots_have_different_pilots() {
        for slot in 0..300 {
            let (a, b) = (pilots(slot), pilots(slot + 1));
            let same = a.iter().zip(&b).filter(|(x, y)| x == y).count();
            assert!(same < PILOT, "slots {slot} and {} share their pilots", slot + 1);
        }
    }

    #[test]
    fn every_mode_fits_a_codeword_and_round_trips() {
        for modulation in Modulation::ALL {
            for rate in Rate::ALL {
                let g = Geometry::new(modulation, rate);
                assert!(g.coded <= g.channel_bits && g.channel_bits - g.coded < 4);
                let payload: Vec<u8> = (0..g.payload).map(|i| (i * 7 + 3) as u8).collect();
                let symbols = g.encode(77, &payload);
                assert_eq!(symbols.len(), crate::profile::CODEWORD_SYMBOLS);
                let mut soft = Vec::new();
                for z in &symbols {
                    modulation.demap(*z, 0.5, &mut soft);
                }
                assert_eq!(g.decode(&soft), Some((77, payload)), "{} {}", modulation.label(), rate.label());
            }
        }
    }

    #[test]
    fn the_modes_carry_what_the_design_says() {
        let payload = |m, r| Geometry::new(m, r).payload;
        assert_eq!(payload(Modulation::Bpsk, Rate::Half), 49);
        assert_eq!(payload(Modulation::Qpsk, Rate::Half), 105);
        assert_eq!(payload(Modulation::Psk8, Rate::ThreeQuarters), 245);
    }

    #[test]
    fn every_symbol_after_the_preamble_is_a_point_of_the_constellation() {
        for modulation in Modulation::ALL {
            let h = Header { modulation, codewords: 2, total: 2, sequence: 0, ..header() };
            let g = h.geometry();
            let cws = vec![g.encode(0, &[1, 2, 3]), g.encode(1, &[4, 5, 6])];
            let points = modulation.constellation();
            for (i, z) in burst(h, &cws).iter().enumerate().skip(PREAMBLE) {
                assert!(points.iter().any(|p| (*p - *z).abs() < 1e-12), "{} symbol {i} is {z:?}", modulation.label());
            }
        }
    }

    #[test]
    fn a_burst_is_as_long_as_the_profile_says() {
        let h = Header { codewords: 2, total: 2, sequence: 0, ..header() };
        let g = h.geometry();
        let cws = vec![g.encode(0, &[]), g.encode(1, &[])];
        assert_eq!(burst(h, &cws).len(), crate::profile::burst_symbols(2, crate::profile::SLOTS_PER_CODEWORD));
    }
}
