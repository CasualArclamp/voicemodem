//! Error control: the checks, the whitening, the convolutional code and the
//! interleaver.
//!
//! The code is the K=7 rate-1/2 pair everybody uses (generators 171 and 133
//! octal), punctured to 2/3 and 3/4 with the usual patterns, decoded with a
//! soft-input Viterbi. It is terminated with six zero tail bits, so the
//! decoder starts and ends in state nought and needs no traceback depth
//! decision: every codeword is decoded whole.
//!
//! Soft values throughout are log-likelihood ratios with the sign convention
//! `log P(0) / P(1)`: positive means a nought, and nought means nothing is
//! known. A punctured bit and an erased slot are both just a nought, which is
//! why erasing a stretch the framer knows is garbage is cheap and effective.

/// CRC-16/CCITT-FALSE: polynomial 0x1021, initial 0xFFFF, no reflection.
pub fn crc16(data: &[u8]) -> u16 {
    let mut crc: u16 = 0xFFFF;
    for &byte in data {
        crc ^= u16::from(byte) << 8;
        for _ in 0..8 {
            crc = if crc & 0x8000 != 0 { (crc << 1) ^ 0x1021 } else { crc << 1 };
        }
    }
    crc
}

/// CRC-32 as in Ethernet and zip: reflected 0xEDB88320, initial and final
/// inversion.
pub fn crc32(data: &[u8]) -> u32 {
    let mut crc: u32 = 0xFFFF_FFFF;
    for &byte in data {
        crc ^= u32::from(byte);
        for _ in 0..8 {
            crc = if crc & 1 != 0 { (crc >> 1) ^ 0xEDB8_8320 } else { crc >> 1 };
        }
    }
    !crc
}

/// Additive whitening from a 15-bit maximal-length register, x^15 + x^14 + 1.
///
/// Applied to every codeword's bytes before coding, so that a block of
/// zeros -- the padding at the end of a short final block, most often --
/// still goes out as symbols that change. A run of identical symbols gives
/// the timing loop nothing to lock to and makes the constellation a dot.
/// The register starts from the same state for every codeword, so it is
/// its own inverse and needs no synchronising.
pub fn whiten(bytes: &mut [u8]) {
    let mut state: u16 = 0x4A80;
    for byte in bytes {
        let mut mask = 0u8;
        for bit in 0..8 {
            let out = ((state >> 14) ^ (state >> 13)) & 1;
            state = ((state << 1) | out) & 0x7FFF;
            mask |= (out as u8) << bit;
        }
        *byte ^= mask;
    }
}

/// [`whiten`] on bits rather than bytes: the same sequence, bit for bit, as
/// [`whiten`] applies to the bytes those bits come from by [`bits_of`].
pub fn whiten_bits(bits: &mut [u8]) {
    let mut state: u16 = 0x4A80;
    for bit in bits {
        let out = ((state >> 14) ^ (state >> 13)) & 1;
        state = ((state << 1) | out) & 0x7FFF;
        *bit ^= out as u8;
    }
}

/// Bits of a byte slice, least significant bit of each byte first.
pub fn bits_of(bytes: &[u8]) -> Vec<u8> {
    bytes.iter().flat_map(|b| (0..8).map(move |i| (b >> i) & 1)).collect()
}

/// Bytes from bits, least significant first; a short final byte is padded
/// with noughts.
pub fn bytes_of(bits: &[u8]) -> Vec<u8> {
    bits.chunks(8).map(|chunk| chunk.iter().enumerate().fold(0u8, |acc, (i, &b)| acc | ((b & 1) << i))).collect()
}

/// Tail bits that bring the encoder back to state nought.
pub const TAIL: usize = 6;

/// The two generators with the newest bit as bit nought: 171 octal taps the
/// current bit and those one, two, three and six back; 133 the current bit
/// and those two, three, five and six back.
const G1: u8 = 0b100_1111;
const G2: u8 = 0b110_1101;
const STATES: usize = 64;

fn parity(x: u8) -> u8 {
    (x.count_ones() & 1) as u8
}

/// The two code bits leaving the encoder when `input` arrives in `state`.
fn outputs(state: usize, input: u8) -> (u8, u8) {
    let reg = (((state << 1) | usize::from(input)) & 0x7F) as u8;
    (parity(reg & G1), parity(reg & G2))
}

/// The code rate, as punctured from the mother rate-1/2 code.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Rate {
    Half,
    TwoThirds,
    ThreeQuarters,
}

impl Rate {
    pub const ALL: [Rate; 3] = [Rate::Half, Rate::TwoThirds, Rate::ThreeQuarters];

    /// Which of the mother code's bits, taken in pairs (first generator, then
    /// second) for each input bit of a period, are sent.
    fn pattern(self) -> &'static [bool] {
        match self {
            Rate::Half => &[true, true],
            // X 11 / Y 10: sends X1 Y1 X2.
            Rate::TwoThirds => &[true, true, true, false],
            // X 101 / Y 110: sends X1 Y1 Y2 X3.
            Rate::ThreeQuarters => &[true, true, false, true, true, false],
        }
    }

    /// Input bits a puncturing period covers, and code bits it sends.
    pub fn period(self) -> (usize, usize) {
        let pattern = self.pattern();
        (pattern.len() / 2, pattern.iter().filter(|k| **k).count())
    }

    pub fn code(self) -> u8 {
        match self {
            Rate::Half => 0,
            Rate::TwoThirds => 1,
            Rate::ThreeQuarters => 2,
        }
    }

    pub fn from_code(code: u8) -> Option<Self> {
        match code {
            0 => Some(Rate::Half),
            1 => Some(Rate::TwoThirds),
            2 => Some(Rate::ThreeQuarters),
            _ => None,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Rate::Half => "1/2",
            Rate::TwoThirds => "2/3",
            Rate::ThreeQuarters => "3/4",
        }
    }

    pub fn parse(text: &str) -> Option<Self> {
        match text.trim() {
            "1/2" | "12" | "half" => Some(Rate::Half),
            "2/3" | "23" => Some(Rate::TwoThirds),
            "3/4" | "34" => Some(Rate::ThreeQuarters),
            _ => None,
        }
    }
}

/// Encode `bits` -- which must already end in [`TAIL`] noughts, and be a
/// whole number of puncturing periods long -- and puncture to `rate`.
pub fn encode(bits: &[u8], rate: Rate) -> Vec<u8> {
    let pattern = rate.pattern();
    let (per, _) = rate.period();
    debug_assert!(bits.len().is_multiple_of(per), "{} bits is not whole periods of {per}", bits.len());
    let mut state = 0usize;
    let mut out = Vec::with_capacity(bits.len() * 2);
    for (i, &bit) in bits.iter().enumerate() {
        let (a, b) = outputs(state, bit & 1);
        state = ((state << 1) | usize::from(bit & 1)) & (STATES - 1);
        let at = 2 * (i % per);
        if pattern[at] {
            out.push(a);
        }
        if pattern[at + 1] {
            out.push(b);
        }
    }
    out
}

/// Code bits sent for `inputs` input bits at `rate`.
pub fn coded_len(inputs: usize, rate: Rate) -> usize {
    let (per, sent) = rate.period();
    inputs / per * sent
}

/// Decode `soft` values of a punctured, terminated codeword of `inputs`
/// input bits (tail included) back to its input bits, tail included.
pub fn decode(soft: &[f64], inputs: usize, rate: Rate) -> Vec<u8> {
    let pattern = rate.pattern();
    let (per, _) = rate.period();
    // Put the punctured positions back as noughts: known to be unknown.
    let mut pairs = Vec::with_capacity(inputs);
    let mut next = soft.iter().copied();
    for i in 0..inputs {
        let at = 2 * (i % per);
        let a = if pattern[at] { next.next().unwrap_or(0.0) } else { 0.0 };
        let b = if pattern[at + 1] { next.next().unwrap_or(0.0) } else { 0.0 };
        pairs.push((a, b));
    }

    // Each branch's metric is the correlation of its code bits with the
    // soft values: +llr for a nought, -llr for a one. Largest wins.
    let mut metric = [f64::NEG_INFINITY; STATES];
    metric[0] = 0.0;
    let mut decisions: Vec<u64> = Vec::with_capacity(inputs);
    let mut next_metric = [f64::NEG_INFINITY; STATES];
    for &(la, lb) in &pairs {
        let mut chosen = 0u64;
        for (n, slot) in next_metric.iter_mut().enumerate() {
            let input = (n & 1) as u8;
            let low = n >> 1;
            let high = low | (STATES >> 1);
            let branch = |from: usize| {
                let (a, b) = outputs(from, input);
                let ma = if a == 0 { la } else { -la };
                let mb = if b == 0 { lb } else { -lb };
                metric[from] + ma + mb
            };
            let (m0, m1) = (branch(low), branch(high));
            if m1 > m0 {
                *slot = m1;
                chosen |= 1 << n;
            } else {
                *slot = m0;
            }
        }
        metric = next_metric;
        decisions.push(chosen);
    }

    // Terminated: the path ends in state nought.
    let mut state = 0usize;
    let mut out = vec![0u8; inputs];
    for (i, chosen) in decisions.iter().enumerate().rev() {
        out[i] = (state & 1) as u8;
        let from_high = (chosen >> state) & 1 == 1;
        state = (state >> 1) | if from_high { STATES >> 1 } else { 0 };
    }
    out
}

/// An interleaver over `n` positions: channel position `j` carries code bit
/// `j * step mod n`, with `step` coprime to `n` and near `n` over the golden
/// ratio.
///
/// A run of consecutive channel bits -- a slot the framer had to erase, a
/// burst of noise, the three bits of one 8PSK symbol -- lands on code bits
/// spread right across the codeword, where the Viterbi decoder sees them as
/// scattered and not as a hole.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Interleaver {
    map: Vec<usize>,
}

impl Interleaver {
    pub fn new(n: usize) -> Self {
        let golden = 0.381_966_011_250_105_1;
        let mut step = ((n as f64 * golden).round() as usize).max(1);
        while n > 1 && gcd(step, n) != 1 {
            step += 1;
        }
        Self { map: (0..n).map(|j| (j * step) % n.max(1)).collect() }
    }

    pub fn len(&self) -> usize {
        self.map.len()
    }

    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    /// Code bits in channel order.
    pub fn interleave<T: Copy>(&self, code: &[T]) -> Vec<T> {
        self.map.iter().map(|&i| code[i]).collect()
    }

    /// Channel values back in code order.
    pub fn deinterleave<T: Copy + Default>(&self, channel: &[T]) -> Vec<T> {
        let mut out = vec![T::default(); self.map.len()];
        for (j, &i) in self.map.iter().enumerate() {
            out[i] = channel[j];
        }
        out
    }
}

fn gcd(mut a: usize, mut b: usize) -> usize {
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a
}

#[cfg(test)]
mod tests {
    use super::*;

    fn random_bits(n: usize, seed: u64) -> Vec<u8> {
        let mut x = seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1;
        (0..n)
            .map(|_| {
                x ^= x >> 12;
                x ^= x << 25;
                x ^= x >> 27;
                (x.wrapping_mul(0x2545_f491_4f6c_dd1d) >> 63) as u8
            })
            .collect()
    }

    fn with_tail(mut bits: Vec<u8>, per: usize) -> Vec<u8> {
        bits.extend([0; TAIL]);
        while !bits.len().is_multiple_of(per) {
            bits.push(0);
        }
        bits
    }

    #[test]
    fn crcs_match_their_check_values() {
        assert_eq!(crc16(b"123456789"), 0x29B1);
        assert_eq!(crc32(b"123456789"), 0xCBF4_3926);
    }

    #[test]
    fn whitening_is_its_own_inverse_and_whitens_zeros() {
        let mut bytes = vec![0u8; 64];
        whiten(&mut bytes);
        let ones: u32 = bytes.iter().map(|b| b.count_ones()).sum();
        assert!((200..312).contains(&ones), "{ones} ones in 512 whitened zeros");
        whiten(&mut bytes);
        assert!(bytes.iter().all(|b| *b == 0));
    }

    #[test]
    fn whitening_bits_matches_whitening_bytes() {
        let mut bytes: Vec<u8> = (0..40).map(|i| (i * 37) as u8).collect();
        let mut bits = bits_of(&bytes);
        whiten(&mut bytes);
        whiten_bits(&mut bits);
        assert_eq!(bytes_of(&bits), bytes);
    }

    #[test]
    fn bits_and_bytes_round_trip() {
        let bytes = vec![0x01, 0x80, 0xA5, 0x3C];
        assert_eq!(bytes_of(&bits_of(&bytes)), bytes);
    }

    #[test]
    fn every_rate_decodes_clean_input_exactly() {
        for rate in Rate::ALL {
            let (per, sent) = rate.period();
            let bits = with_tail(random_bits(600, 3), per);
            let code = encode(&bits, rate);
            assert_eq!(code.len(), coded_len(bits.len(), rate));
            assert_eq!(code.len(), bits.len() / per * sent);
            let soft: Vec<f64> = code.iter().map(|&c| if c == 0 { 1.0 } else { -1.0 }).collect();
            assert_eq!(decode(&soft, bits.len(), rate), bits, "rate {}", rate.label());
        }
    }

    #[test]
    fn half_rate_corrects_scattered_hard_errors() {
        let bits = with_tail(random_bits(1000, 7), 1);
        let code = encode(&bits, Rate::Half);
        let mut soft: Vec<f64> = code.iter().map(|&c| if c == 0 { 1.0 } else { -1.0 }).collect();
        // One bit in twenty wrong, never two close together.
        for i in (5..soft.len()).step_by(20) {
            soft[i] = -soft[i];
        }
        assert_eq!(decode(&soft, bits.len(), Rate::Half), bits);
    }

    #[test]
    fn erasures_spread_by_the_interleaver_are_recovered() {
        // A quarter of the channel erased in one run: after deinterleaving it
        // is scattered, and rate one half has the redundancy to fill it.
        let bits = with_tail(random_bits(890, 11), 1);
        let code = encode(&bits, Rate::Half);
        let il = Interleaver::new(code.len());
        let mut channel: Vec<f64> =
            il.interleave(&code).iter().map(|&c| if c == 0 { 1.0 } else { -1.0 }).collect();
        let n = channel.len();
        for v in &mut channel[n / 3..n / 3 + n / 4] {
            *v = 0.0;
        }
        let soft = il.deinterleave(&channel);
        assert_eq!(decode(&soft, bits.len(), Rate::Half), bits);
    }

    #[test]
    fn interleaver_is_a_permutation() {
        for n in [1, 2, 172, 896, 1792, 2688] {
            let il = Interleaver::new(n);
            let mut seen = vec![false; n];
            for &i in &il.map {
                assert!(!seen[i], "position {i} twice for n={n}");
                seen[i] = true;
            }
            let values: Vec<usize> = (0..n).collect();
            assert_eq!(il.deinterleave(&il.interleave(&values)), values);
        }
    }
}
