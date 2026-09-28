//! Speech over the modem: which codec in which mode, how its frames fill a
//! codeword, and how a transmission is paced while someone is talking.
//!
//! A codec is known here only by what the modem needs of it -- bits a frame
//! and how long a frame lasts. The codecs themselves are in the `voice` crate.
//!
//! Every codeword of a voice burst carries whole codec frames:
//!
//! ```text
//! seq 5 | count 6 | end 1 | text 8 | count frames of the codec's bits | pad | CRC-16
//! ```
//!
//! `seq` counts codewords, so a receiver knows how many it missed; `count`
//! says how many of the codeword's frames are in use, because a codeword can
//! hold more speech than it takes to send -- that spare is how the
//! transmitter catches up on the time a preamble took, and how it can end
//! a transmission part way through a codeword; `end` marks the last
//! codeword of a transmission; `text` is one character of a short message
//! sent over and over, a callsign most often, a character a codeword. The
//! CRC decides whether the frames are played or replaced by silence.
//!
//! A transmission is bursts back to back, each with its own preamble, so a
//! receiver that tunes in part way, or loses one to a fade, is back within
//! seconds. The preambles cost air time, which the codewords' spare capacity
//! pays back: [`VoiceMode::codewords_per_burst`] is chosen so that it does.

use std::collections::VecDeque;

use dsp::Complex;

use crate::fec::{self, Rate};
use crate::frame::{Geometry, Header, Kind, codeword_slots, pilot_symbols, preamble};
use crate::profile::{MAX_CODEWORDS, PILOT, PREAMBLE, Profile};
use crate::psk::Modulation;

/// A speech codec, as far as the modem is concerned.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Codec {
    Codec2_3200,
    Codec2_2400,
    Codec2_1600,
    Codec2_1400,
    Codec2_1300,
    Codec2_1200,
    /// Meta's EnCodec, 24 kHz model, two codebooks: 1.5 kbit/s.
    Encodec1500,
    /// EnCodec, four codebooks: 3 kbit/s.
    Encodec3000,
}

impl Codec {
    pub const ALL: [Codec; 8] = [
        Codec::Codec2_3200,
        Codec::Codec2_2400,
        Codec::Codec2_1600,
        Codec::Codec2_1400,
        Codec::Codec2_1300,
        Codec::Codec2_1200,
        Codec::Encodec1500,
        Codec::Encodec3000,
    ];

    /// The four bits the header carries.
    pub fn id(self) -> u8 {
        match self {
            Codec::Codec2_3200 => 0,
            Codec::Codec2_2400 => 1,
            Codec::Codec2_1600 => 2,
            Codec::Codec2_1400 => 3,
            Codec::Codec2_1300 => 4,
            Codec::Codec2_1200 => 5,
            Codec::Encodec1500 => 8,
            Codec::Encodec3000 => 9,
        }
    }

    pub fn from_id(id: u8) -> Option<Self> {
        Self::ALL.into_iter().find(|c| c.id() == id)
    }

    /// Bits a frame.
    pub fn bits(self) -> usize {
        match self {
            Codec::Codec2_3200 | Codec::Codec2_1600 => 64,
            Codec::Codec2_2400 | Codec::Codec2_1200 => 48,
            Codec::Codec2_1400 => 56,
            Codec::Codec2_1300 => 52,
            // Ten bits a codebook at 75 frames a second.
            Codec::Encodec1500 => 20,
            Codec::Encodec3000 => 40,
        }
    }

    /// How long a frame lasts, in seconds.
    pub fn frame_seconds(self) -> f64 {
        match self {
            Codec::Codec2_3200 | Codec::Codec2_2400 => 0.020,
            Codec::Codec2_1600 | Codec::Codec2_1400 | Codec::Codec2_1300 | Codec::Codec2_1200 => 0.040,
            Codec::Encodec1500 | Codec::Encodec3000 => 1.0 / 75.0,
        }
    }

    pub fn bit_rate(self) -> f64 {
        self.bits() as f64 / self.frame_seconds()
    }

    pub fn label(self) -> &'static str {
        match self {
            Codec::Codec2_3200 => "Codec2 3200",
            Codec::Codec2_2400 => "Codec2 2400",
            Codec::Codec2_1600 => "Codec2 1600",
            Codec::Codec2_1400 => "Codec2 1400",
            Codec::Codec2_1300 => "Codec2 1300",
            Codec::Codec2_1200 => "Codec2 1200",
            Codec::Encodec1500 => "EnCodec 1.5k",
            Codec::Encodec3000 => "EnCodec 3k",
        }
    }
}

/// A voice mode: a profile, a modulation and code rate, and a codec.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct VoiceMode {
    pub name: &'static str,
    pub profile: Profile,
    pub modulation: Modulation,
    pub rate: Rate,
    pub codec: Codec,
    /// Slots a codeword spans. Four at either rate is 320 ms narrow and
    /// 213 ms wide: a codeword is sent only once its speech has been spoken
    /// and played only once it has all arrived, so its length comes twice
    /// into the delay. Wide's BPSK needs six to carry its eight frames.
    pub slots: usize,
}

const fn mode(
    name: &'static str,
    profile: Profile,
    modulation: Modulation,
    rate: Rate,
    codec: Codec,
    slots: usize,
) -> VoiceMode {
    VoiceMode { name, profile, modulation, rate, codec, slots }
}

/// Every voice mode, most robust first within each profile.
///
/// Narrow has no BPSK voice mode: at 1600 baud, BPSK nets at most 1050
/// bit/s once the pilots are paid for, and the lowest Codec2 rate available
/// in Rust is 1200. Wide's most robust mode is BPSK.
pub const VOICE_MODES: [VoiceMode; 10] = [
    mode("narrow-robust", Profile::Narrow, Modulation::Qpsk, Rate::Half, Codec::Codec2_1200, 4),
    mode("narrow-standard", Profile::Narrow, Modulation::Qpsk, Rate::TwoThirds, Codec::Codec2_1600, 4),
    mode("narrow-high", Profile::Narrow, Modulation::Psk8, Rate::TwoThirds, Codec::Codec2_2400, 4),
    mode("narrow-neural", Profile::Narrow, Modulation::Qpsk, Rate::TwoThirds, Codec::Encodec1500, 4),
    mode("wide-robust", Profile::Wide, Modulation::Bpsk, Rate::TwoThirds, Codec::Codec2_1200, 6),
    mode("wide-standard", Profile::Wide, Modulation::Qpsk, Rate::Half, Codec::Codec2_1600, 4),
    mode("wide-high", Profile::Wide, Modulation::Qpsk, Rate::ThreeQuarters, Codec::Codec2_2400, 4),
    mode("wide-best", Profile::Wide, Modulation::Psk8, Rate::TwoThirds, Codec::Codec2_3200, 4),
    mode("wide-neural", Profile::Wide, Modulation::Qpsk, Rate::Half, Codec::Encodec1500, 4),
    mode("wide-neural-hq", Profile::Wide, Modulation::Psk8, Rate::ThreeQuarters, Codec::Encodec3000, 4),
];

/// The longest a voice burst is let run between preambles: a receiver that
/// tunes in late, or loses a burst to a fade, waits no longer than this.
const BURST_SECONDS: f64 = 5.0;

/// Bits of every voice codeword that are not speech: sequence, count, end,
/// text and CRC.
pub const OVERHEAD: usize = 5 + 6 + 1 + 8 + 16;

impl VoiceMode {
    pub fn by_name(name: &str) -> Option<&'static VoiceMode> {
        VOICE_MODES.iter().find(|m| m.name.eq_ignore_ascii_case(name.trim()))
    }

    /// The mode a header announces, heard on `profile`.
    pub fn of(profile: Profile, header: &Header) -> Option<&'static VoiceMode> {
        let codec = Codec::from_id(header.codec)?;
        VOICE_MODES.iter().find(|m| {
            m.profile == profile && m.modulation == header.modulation && m.rate == header.rate && m.codec == codec
        })
    }

    /// The most robust mode of `profile`: what a transmitter starts with.
    pub fn robust(profile: Profile) -> &'static VoiceMode {
        VOICE_MODES.iter().find(|m| m.profile == profile).expect("every profile has a voice mode")
    }

    /// The modes of `profile`, most robust first.
    pub fn of_profile(profile: Profile) -> impl Iterator<Item = &'static VoiceMode> {
        VOICE_MODES.iter().filter(move |m| m.profile == profile)
    }

    pub fn geometry(&self) -> Geometry {
        Geometry::with_slots(self.modulation, self.rate, self.slots)
    }

    /// Codec frames a codeword holds.
    pub fn frames_per_codeword(&self) -> usize {
        (self.geometry().info_bits() - OVERHEAD) / self.codec.bits()
    }

    /// Seconds a codeword takes to send.
    pub fn codeword_seconds(&self) -> f64 {
        self.geometry().seconds(self.profile)
    }

    /// Seconds of speech a full codeword holds.
    pub fn codeword_speech(&self) -> f64 {
        self.frames_per_codeword() as f64 * self.codec.frame_seconds()
    }

    /// Codewords a burst: enough that the speech the codewords can hold
    /// beyond their own air time pays back the preamble's with half as much
    /// again to spare, so the transmitter does not fall behind the speaker;
    /// but no more than [`BURST_SECONDS`] of them.
    ///
    /// A mode whose codewords hold only their own air time's speech -- the
    /// robust ones, which have the least to spare -- cannot pay a preamble
    /// back that way. Its transmitter drops frames of silence instead
    /// ([`VoiceTx`]), which only shortens the pauses between words.
    pub fn codewords_per_burst(&self) -> usize {
        let longest = ((BURST_SECONDS / self.codeword_seconds()) as usize).clamp(4, MAX_CODEWORDS);
        let spare = self.codeword_speech() - self.codeword_seconds();
        let preamble = self.profile.seconds(PREAMBLE + PILOT);
        if spare <= 1e-9 {
            return longest;
        }
        ((1.5 * preamble / spare).ceil() as usize).clamp(4, longest)
    }

    /// The header of burst `sequence` of transmission `stream`.
    pub fn header(&self, stream: u16, sequence: u16) -> Header {
        Header {
            kind: Kind::Voice,
            modulation: self.modulation,
            rate: self.rate,
            codec: self.codec.id(),
            codewords: self.codewords_per_burst() as u8,
            stream,
            sequence,
            total: 0,
        }
    }

    /// A codeword's data symbols.
    pub fn encode(&self, codeword: &VoiceCodeword) -> Vec<Complex> {
        let frames = self.frames_per_codeword();
        assert!(codeword.frames.len() <= frames, "{} frames into a codeword of {frames}", codeword.frames.len());
        let geometry = self.geometry();
        let mut bits = Vec::with_capacity(geometry.info_bits());
        push_bits(&mut bits, u64::from(codeword.seq & 0x1F), 5);
        push_bits(&mut bits, codeword.frames.len() as u64, 6);
        push_bits(&mut bits, u64::from(codeword.end), 1);
        push_bits(&mut bits, u64::from(codeword.text), 8);
        for frame in &codeword.frames {
            assert_eq!(frame.len(), self.codec.bits(), "a {} frame", self.codec.label());
            bits.extend(frame.iter().map(|b| b & 1));
        }
        bits.resize(geometry.info_bits() - 16, 0);
        let crc = fec::crc16(&fec::bytes_of(&bits));
        push_bits(&mut bits, u64::from(crc), 16);
        geometry.encode_bits(&bits)
    }

    /// A codeword from the soft values of its channel bits, if its CRC holds.
    pub fn decode(&self, soft: &[f64]) -> Option<VoiceCodeword> {
        let geometry = self.geometry();
        let bits = geometry.decode_bits(soft)?;
        let (body, crc) = bits.split_at(geometry.info_bits() - 16);
        if fec::crc16(&fec::bytes_of(body)) != read_bits(crc) as u16 {
            return None;
        }
        let count = read_bits(&body[5..11]) as usize;
        if count > self.frames_per_codeword() {
            return None;
        }
        let b = self.codec.bits();
        let frames = (0..count).map(|i| body[20 + i * b..20 + (i + 1) * b].to_vec()).collect();
        Some(VoiceCodeword {
            seq: read_bits(&body[..5]) as u8,
            frames,
            end: body[11] == 1,
            text: read_bits(&body[12..20]) as u8,
        })
    }
}

fn push_bits(bits: &mut Vec<u8>, value: u64, n: usize) {
    for i in (0..n).rev() {
        bits.push(((value >> i) & 1) as u8);
    }
}

fn read_bits(bits: &[u8]) -> u64 {
    bits.iter().fold(0, |acc, &b| (acc << 1) | u64::from(b & 1))
}

/// What a voice codeword carries.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VoiceCodeword {
    /// Codewords sent before this one in the transmission, modulo 32.
    pub seq: u8,
    /// Codec frames, each a vector of its bits.
    pub frames: Vec<Vec<u8>>,
    /// The last codeword of the transmission.
    pub end: bool,
    /// One character of the transmission's text, or nought.
    pub text: u8,
}

/// A transmission being sent: codec frames in as they are spoken, symbols
/// out as the line wants them.
///
/// Every preamble is air time with no speech in it, so after one the
/// transmitter is that far behind the speaker. A mode whose codewords hold
/// more speech than their air time catches up by itself. One whose codewords
/// hold only their own air time's worth -- the robust modes -- catches up by
/// dropping frames the talker marked as silence: a pause between words comes
/// out a little shorter, and nothing that was said is lost. Past a cap, if
/// one is set, the oldest frames go whatever they are, so that someone who
/// talks without drawing breath is not heard further and further behind.
#[derive(Debug, Clone)]
pub struct VoiceTx {
    mode: &'static VoiceMode,
    stream: u16,
    /// Burst number, codewords sent in this burst, and whether its preamble
    /// has gone.
    burst: u16,
    in_burst: usize,
    preamble_sent: bool,
    seq: u8,
    /// Frames waiting, each with whether it is silence.
    frames: VecDeque<(Vec<u8>, bool)>,
    cap: Option<usize>,
    /// Frames of silence dropped to catch up, and frames dropped at the cap.
    trimmed: usize,
    dropped: usize,
    text: Vec<u8>,
    text_at: usize,
    ending: bool,
    done: bool,
}

impl VoiceTx {
    /// A transmission in `mode`, known as `stream`, sending `text` a
    /// character a codeword, over and over.
    pub fn new(mode: &'static VoiceMode, stream: u16, text: &str) -> Self {
        let text: Vec<u8> = text.bytes().filter(|b| b.is_ascii_graphic() || *b == b' ').take(64).collect();
        Self {
            mode,
            stream,
            burst: 0,
            in_burst: 0,
            preamble_sent: false,
            seq: 0,
            frames: VecDeque::new(),
            cap: None,
            trimmed: 0,
            dropped: 0,
            text,
            text_at: 0,
            ending: false,
            done: false,
        }
    }

    /// Never let more than `frames` frames wait: past that the oldest are
    /// dropped, speech or not. For a live talker; a recording being sent
    /// wants every frame, however far ahead of the line it is.
    pub fn with_cap(mut self, frames: usize) -> Self {
        self.cap = Some(frames.max(self.mode.frames_per_codeword()));
        self
    }

    pub fn mode(&self) -> &'static VoiceMode {
        self.mode
    }

    /// A codec frame, as its bits.
    pub fn push_frame(&mut self, bits: Vec<u8>) {
        self.push_spoken(bits, false);
    }

    /// A codec frame, and whether it is silence that may be dropped to
    /// catch up.
    pub fn push_spoken(&mut self, bits: Vec<u8>, quiet: bool) {
        if !self.ending {
            self.frames.push_back((bits, quiet));
        }
    }

    /// Frames of silence dropped to catch up, and frames of anything dropped
    /// at the cap.
    pub fn trimmed(&self) -> (usize, usize) {
        (self.trimmed, self.dropped)
    }

    /// Catch up on the speaker: silence first, oldest first, down to a
    /// codeword's worth -- what a transmitter keeping up has waiting when it
    /// makes a codeword; then, past the cap, whatever is oldest.
    fn catch_up(&mut self) {
        let mut over = self.frames.len().saturating_sub(self.mode.frames_per_codeword());
        let mut i = 0;
        while over > 0 && i < self.frames.len() {
            if self.frames[i].1 {
                self.frames.remove(i);
                self.trimmed += 1;
                over -= 1;
            } else {
                i += 1;
            }
        }
        if let Some(cap) = self.cap {
            while self.frames.len() > cap {
                self.frames.pop_front();
                self.dropped += 1;
            }
        }
    }

    /// Frames waiting to go.
    pub fn waiting(&self) -> usize {
        self.frames.len()
    }

    /// The speaker has stopped: send what is left and end.
    pub fn end(&mut self) {
        self.ending = true;
    }

    pub fn is_done(&self) -> bool {
        self.done
    }

    /// The next symbols to send: a preamble, or a codeword with its slots'
    /// pilots, and after the last codeword of a burst the closing pilots.
    /// None once the transmission is over.
    pub fn next_symbols(&mut self) -> Option<Vec<Complex>> {
        if self.done {
            return None;
        }
        let per_burst = self.mode.codewords_per_burst();
        if !self.preamble_sent {
            self.preamble_sent = true;
            return Some(preamble(self.mode.header(self.stream, self.burst)));
        }
        self.catch_up();
        let take = self.mode.frames_per_codeword().min(self.frames.len());
        let frames: Vec<Vec<u8>> = self.frames.drain(..take).map(|(bits, _)| bits).collect();
        let end = self.ending && self.frames.is_empty();
        let text = if self.text.is_empty() {
            0
        } else {
            let c = self.text[self.text_at % self.text.len()];
            self.text_at += 1;
            c
        };
        let codeword = VoiceCodeword { seq: self.seq & 0x1F, frames, end, text };
        self.seq = self.seq.wrapping_add(1);
        let mut symbols = codeword_slots(self.in_burst * self.mode.slots, &self.mode.encode(&codeword));
        self.in_burst += 1;
        if end || self.in_burst == per_burst {
            symbols.extend(pilot_symbols(self.in_burst * self.mode.slots));
            self.in_burst = 0;
            self.burst = self.burst.wrapping_add(1);
            self.preamble_sent = false;
            self.done = end;
        }
        Some(symbols)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_mode_keeps_up_with_its_speaker() {
        for mode in &VOICE_MODES {
            let frames = mode.frames_per_codeword();
            let n = mode.codewords_per_burst();
            let air = n as f64 * mode.codeword_seconds() + mode.profile.seconds(PREAMBLE + PILOT);
            let speech = n as f64 * mode.codeword_speech();
            assert!((1..64).contains(&frames), "{}: {frames} frames", mode.name);
            // Every codeword holds at least its own air time's speech...
            assert!(mode.codeword_speech() >= mode.codeword_seconds() - 1e-9, "{}", mode.name);
            // ...and the preambles are paid back by the spare, or else by at
            // most a twentieth of the speech being silence the transmitter
            // can drop -- ordinary speech pauses for far more than that --
            // in bursts no longer than five seconds.
            let short = (air - speech).max(0.0) / air;
            assert!(short <= 0.05, "{}: {:.1}% of the air time left unpaid", mode.name, 100.0 * short);
            assert!(air <= BURST_SECONDS + mode.profile.seconds(PREAMBLE + PILOT) + 1e-9, "{}: {air:.1} s bursts", mode.name);
        }
    }

    #[test]
    fn half_second_codewords_or_shorter() {
        for mode in &VOICE_MODES {
            assert!(mode.codeword_seconds() <= 0.33, "{}: {:.3} s codewords", mode.name, mode.codeword_seconds());
        }
    }

    #[test]
    fn a_transmitter_behind_drops_silence_and_keeps_speech() {
        let mode = VoiceMode::robust(Profile::Narrow);
        let per = mode.frames_per_codeword();
        let mut tx = VoiceTx::new(mode, 1, "");
        // A codeword's worth of speech, then five frames of silence and five
        // of speech: five more than a transmitter keeping up has waiting.
        for i in 0..per + 10 {
            let quiet = (per..per + 5).contains(&i);
            tx.push_spoken(vec![u8::from(quiet); mode.codec.bits()], quiet);
        }
        tx.next_symbols(); // the preamble
        tx.next_symbols(); // a codeword, made after catching up
        assert_eq!(tx.trimmed(), (5, 0));
        assert_eq!(tx.waiting(), 10 - 5 - (per - per.min(per)));
        assert!(tx.frames.iter().all(|(_, quiet)| !quiet), "silence left waiting");
    }

    #[test]
    fn past_the_cap_the_oldest_go_whatever_they_are() {
        let mode = VoiceMode::robust(Profile::Narrow);
        let per = mode.frames_per_codeword();
        let mut tx = VoiceTx::new(mode, 1, "").with_cap(2 * per);
        for _ in 0..5 * per {
            tx.push_frame(vec![1; mode.codec.bits()]);
        }
        tx.next_symbols();
        tx.next_symbols();
        assert_eq!(tx.trimmed(), (0, 3 * per));
        assert_eq!(tx.waiting(), per);
    }

    #[test]
    fn modes_are_found_by_name_and_by_header() {
        for mode in &VOICE_MODES {
            assert_eq!(VoiceMode::by_name(mode.name), Some(mode));
            assert_eq!(VoiceMode::of(mode.profile, &mode.header(1, 2)), Some(mode));
        }
        assert_eq!(VoiceMode::robust(Profile::Wide).modulation, Modulation::Bpsk);
    }

    #[test]
    fn a_codeword_round_trips() {
        for mode in &VOICE_MODES {
            let b = mode.codec.bits();
            let frames: Vec<Vec<u8>> =
                (0..mode.frames_per_codeword() - 1).map(|f| (0..b).map(|i| ((i * 3 + f) % 5 == 0) as u8).collect()).collect();
            let codeword = VoiceCodeword { seq: 17, frames, end: true, text: b'K' };
            let mut soft = Vec::new();
            for z in mode.encode(&codeword) {
                mode.modulation.demap(z, 0.5, &mut soft);
            }
            assert_eq!(mode.decode(&soft), Some(codeword), "{}", mode.name);
        }
    }

    #[test]
    fn a_transmission_ends_where_it_is_told() {
        let mode = VoiceMode::robust(Profile::Narrow);
        let mut tx = VoiceTx::new(mode, 5, "VK2XYZ");
        let mut sent = Vec::new();
        for _ in 0..mode.frames_per_codeword() * 2 + 3 {
            tx.push_frame(vec![1; mode.codec.bits()]);
        }
        tx.end();
        while let Some(symbols) = tx.next_symbols() {
            sent.push(symbols.len());
        }
        // A preamble, three codewords, the last with the closing pilots.
        let codeword = mode.slots * crate::profile::SLOT;
        assert_eq!(sent, vec![PREAMBLE, codeword, codeword, codeword + PILOT]);
        assert!(tx.is_done());
    }
}
