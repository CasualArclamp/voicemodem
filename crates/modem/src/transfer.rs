//! What the codewords carry: a text message or a file, cut into numbered
//! blocks, and put back together at the far end.
//!
//! A transfer is one byte stream -- a short description followed by the
//! content -- cut into blocks of whatever a codeword holds in the chosen
//! mode. Block 0 begins with the description, so a short message is one
//! codeword and nothing else. The stream carries its own length and CRC-32,
//! checked when the last block is in, on top of every block's own CRC.
//!
//! ```text
//! "VM" | version | kind | length u32 | crc32 u32 | name length | name | content
//! ```

use crate::fec::{Rate, crc32};
use crate::frame::{Geometry, Header, Kind};
use crate::profile::{MAX_CODEWORDS, Profile};
use crate::psk::Modulation;

const MAGIC: &[u8; 2] = b"VM";
const VERSION: u8 = 1;
const FIXED: usize = 2 + 1 + 1 + 4 + 4 + 1;

/// What a transfer carries.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Content {
    Text(String),
    File { name: String, data: Vec<u8> },
}

impl Content {
    fn kind(&self) -> u8 {
        match self {
            Content::Text(_) => 0,
            Content::File { .. } => 1,
        }
    }

    fn body(&self) -> &[u8] {
        match self {
            Content::Text(text) => text.as_bytes(),
            Content::File { data, .. } => data,
        }
    }

    fn name(&self) -> &str {
        match self {
            Content::Text(_) => "",
            Content::File { name, .. } => name,
        }
    }

    /// The stream: the description and then the content.
    pub fn to_stream(&self) -> Vec<u8> {
        let body = self.body();
        let name = truncate(self.name(), 255);
        let mut out = Vec::with_capacity(FIXED + name.len() + body.len());
        out.extend_from_slice(MAGIC);
        out.push(VERSION);
        out.push(self.kind());
        out.extend_from_slice(&u32::try_from(body.len()).unwrap_or(u32::MAX).to_le_bytes());
        out.extend_from_slice(&crc32(body).to_le_bytes());
        out.push(name.len() as u8);
        out.extend_from_slice(name.as_bytes());
        out.extend_from_slice(body);
        out
    }

    /// Content from a stream, which may run on past its end with padding.
    pub fn from_stream(stream: &[u8]) -> Result<Self, String> {
        if stream.len() < FIXED || &stream[..2] != MAGIC {
            return Err("not a voicemodem transfer".into());
        }
        if stream[2] != VERSION {
            return Err(format!("transfer version {} is not understood", stream[2]));
        }
        let kind = stream[3];
        let length = u32::from_le_bytes(stream[4..8].try_into().map_err(|_| "short")?) as usize;
        let crc = u32::from_le_bytes(stream[8..12].try_into().map_err(|_| "short")?);
        let name_len = usize::from(stream[12]);
        let body_at = FIXED + name_len;
        if stream.len() < body_at + length {
            return Err(format!("{} bytes of a {length}-byte transfer", stream.len().saturating_sub(body_at)));
        }
        let name = String::from_utf8_lossy(&stream[FIXED..body_at]).into_owned();
        let body = &stream[body_at..body_at + length];
        if crc32(body) != crc {
            return Err("the transfer's CRC does not match".into());
        }
        match kind {
            0 => Ok(Content::Text(String::from_utf8_lossy(body).into_owned())),
            1 => Ok(Content::File { name, data: body.to_vec() }),
            other => Err(format!("unknown transfer kind {other}")),
        }
    }
}

/// `text` cut to at most `max` bytes on a character boundary.
fn truncate(text: &str, max: usize) -> &str {
    if text.len() <= max {
        return text;
    }
    let mut end = max;
    while !text.is_char_boundary(end) {
        end -= 1;
    }
    &text[..end]
}

/// A transfer ready to send.
#[derive(Debug, Clone)]
pub struct Outgoing {
    pub id: u16,
    pub geometry: Geometry,
    stream: Vec<u8>,
    /// Codewords a burst, at most.
    per_burst: usize,
}

impl Outgoing {
    /// Cut `content` into blocks for `modulation` at `rate`.
    pub fn new(content: &Content, id: u16, modulation: Modulation, rate: Rate) -> Result<Self, String> {
        let geometry = Geometry::new(modulation, rate);
        let stream = content.to_stream();
        let blocks = stream.len().div_ceil(geometry.payload);
        if blocks > usize::from(u16::MAX) {
            let most = usize::from(u16::MAX) * geometry.payload - FIXED - 255;
            return Err(format!(
                "{} bytes is more than {} {} carries in one transfer ({most} bytes)",
                content.body().len(),
                modulation.label(),
                rate.label(),
            ));
        }
        Ok(Self { id, geometry, stream, per_burst: MAX_CODEWORDS })
    }

    /// Send at most `codewords` a burst, which must be between one and
    /// [`MAX_CODEWORDS`].
    pub fn with_burst_length(mut self, codewords: usize) -> Self {
        self.per_burst = codewords.clamp(1, MAX_CODEWORDS);
        self
    }

    pub fn blocks(&self) -> usize {
        self.stream.len().div_ceil(self.geometry.payload)
    }

    pub fn bursts(&self) -> usize {
        self.blocks().div_ceil(self.per_burst)
    }

    /// Bytes on the line, the description included.
    pub fn len(&self) -> usize {
        self.stream.len()
    }

    pub fn is_empty(&self) -> bool {
        self.stream.is_empty()
    }

    /// Block `index`'s payload.
    pub fn block(&self, index: usize) -> &[u8] {
        let from = index * self.geometry.payload;
        let to = (from + self.geometry.payload).min(self.stream.len());
        &self.stream[from.min(to)..to]
    }

    /// The header and symbols of burst `burst`, which carries blocks from
    /// `burst * per_burst` on.
    pub fn burst(&self, burst: usize) -> (Header, Vec<dsp::Complex>) {
        let first = burst * self.per_burst;
        let count = (self.blocks() - first).min(self.per_burst);
        self.burst_of(first, count)
    }

    /// The header and symbols of a burst carrying `count` blocks from
    /// `first` -- for sending some blocks again.
    pub fn burst_of(&self, first: usize, count: usize) -> (Header, Vec<dsp::Complex>) {
        let header = Header {
            kind: Kind::Data,
            modulation: self.geometry.modulation,
            rate: self.geometry.rate,
            codec: 0,
            codewords: count as u8,
            stream: self.id,
            sequence: first as u16,
            total: self.blocks() as u16,
        };
        let codewords: Vec<Vec<dsp::Complex>> =
            (first..first + count).map(|i| self.geometry.encode(i as u16, self.block(i))).collect();
        (header, crate::frame::burst(header, &codewords))
    }

    /// Seconds the whole transfer takes on the line in `profile`.
    pub fn seconds(&self, profile: Profile) -> f64 {
        let blocks = self.blocks();
        let full = blocks / self.per_burst;
        let rest = blocks % self.per_burst;
        full as f64 * profile.burst_seconds(self.per_burst, self.geometry.slots)
            + if rest > 0 { profile.burst_seconds(rest, self.geometry.slots) } else { 0.0 }
    }
}

/// A transfer that arrived whole, or as whole as it is going to.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Delivery {
    pub transfer: u16,
    pub content: Content,
}

/// A transfer being put back together.
#[derive(Debug, Clone)]
struct Incoming {
    id: u16,
    payload: usize,
    blocks: Vec<Option<Vec<u8>>>,
    delivered: bool,
    /// When it was last added to, in the assembler's count of blocks.
    touched: u64,
}

/// Transfers being put back together, the most recent few.
#[derive(Debug, Clone, Default)]
pub struct Assembler {
    incoming: Vec<Incoming>,
    clock: u64,
}

/// How many transfers are remembered at once.
const REMEMBERED: usize = 8;

impl Assembler {
    /// Add block `index` of the transfer `header` belongs to. The transfer, if
    /// this completed it; an error if it completed and did not check.
    pub fn add(&mut self, header: &Header, index: u16, data: Vec<u8>) -> Option<Result<Delivery, String>> {
        self.clock += 1;
        let payload = header.geometry().payload;
        let total = usize::from(header.total);
        let at = match self.incoming.iter().position(|t| t.id == header.stream && t.blocks.len() == total && t.payload == payload) {
            Some(at) => at,
            None => {
                if self.incoming.len() >= REMEMBERED {
                    let oldest = self.incoming.iter().enumerate().min_by_key(|(_, t)| t.touched).map_or(0, |(i, _)| i);
                    self.incoming.remove(oldest);
                }
                self.incoming.push(Incoming { id: header.stream, payload, blocks: vec![None; total], delivered: false, touched: 0 });
                self.incoming.len() - 1
            }
        };
        let transfer = &mut self.incoming[at];
        transfer.touched = self.clock;
        let slot = transfer.blocks.get_mut(usize::from(index))?;
        if slot.is_none() {
            *slot = Some(data);
        }
        if transfer.delivered || transfer.blocks.iter().any(Option::is_none) {
            return None;
        }
        transfer.delivered = true;
        let stream: Vec<u8> = transfer.blocks.iter().flatten().flatten().copied().collect();
        Some(Content::from_stream(&stream).map(|content| Delivery { transfer: header.stream, content }))
    }

    /// Blocks in and blocks expected of transfer `id`, if it is remembered.
    pub fn progress(&self, id: u16) -> Option<(usize, usize)> {
        let t = self.incoming.iter().filter(|t| t.id == id).max_by_key(|t| t.touched)?;
        Some((t.blocks.iter().filter(|b| b.is_some()).count(), t.blocks.len()))
    }

    /// The blocks of transfer `id` still missing, if it is remembered.
    pub fn missing(&self, id: u16) -> Option<Vec<u16>> {
        let t = self.incoming.iter().filter(|t| t.id == id).max_by_key(|t| t.touched)?;
        Some(t.blocks.iter().enumerate().filter(|(_, b)| b.is_none()).map(|(i, _)| i as u16).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn streams_round_trip() {
        for content in [
            Content::Text("hello, line".into()),
            Content::Text(String::new()),
            Content::File { name: "photo.jpg".into(), data: (0..3000).map(|i| (i * 31) as u8).collect() },
        ] {
            let mut stream = content.to_stream();
            stream.extend([0u8; 17]);
            assert_eq!(Content::from_stream(&stream), Ok(content));
        }
    }

    #[test]
    fn a_damaged_stream_is_refused() {
        let mut stream = Content::Text("hello".into()).to_stream();
        let last = stream.len() - 1;
        stream[last] ^= 1;
        assert!(Content::from_stream(&stream).is_err());
    }

    #[test]
    fn blocks_reassemble_in_any_order() {
        let content = Content::File { name: "a.bin".into(), data: (0..1000).map(|i| (i % 251) as u8).collect() };
        let out = Outgoing::new(&content, 7, Modulation::Qpsk, Rate::Half).unwrap().with_burst_length(4);
        assert_eq!(out.blocks(), 10);
        assert_eq!(out.bursts(), 3);
        let mut assembler = Assembler::default();
        let mut delivered = None;
        for i in [3, 0, 9, 1, 2, 8, 4, 5, 6, 7] {
            let burst = i / 4;
            let (header, _) = out.burst(burst);
            if let Some(d) = assembler.add(&header, i as u16, out.block(i).to_vec()) {
                delivered = Some(d);
            }
        }
        assert_eq!(delivered, Some(Ok(Delivery { transfer: 7, content })));
    }

    #[test]
    fn a_missing_block_is_reported() {
        let content = Content::Text("x".repeat(300));
        let out = Outgoing::new(&content, 9, Modulation::Bpsk, Rate::Half).unwrap();
        let (header, _) = out.burst(0);
        let mut assembler = Assembler::default();
        for i in (0..out.blocks()).filter(|i| *i != 2) {
            assert!(assembler.add(&header, i as u16, out.block(i).to_vec()).is_none());
        }
        assert_eq!(assembler.missing(9), Some(vec![2]));
        assert_eq!(assembler.progress(9), Some((out.blocks() - 1, out.blocks())));
    }
}
