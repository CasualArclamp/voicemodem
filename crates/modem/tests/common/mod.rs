//! What the integration tests share: a transmitter's samples, a receiver
//! taking them one at a time as it would from a sound card, and a readable
//! account of what it made of them.
//!
//! `cargo test --release -p modem -- --nocapture` prints, for each burst,
//! what it trained to and how it went.

#![allow(dead_code)]

use modem::profile::FS;
use modem::{Event, Modulator, Profile, Receiver};

/// Seconds of silence before the first burst and after the last.
pub const LEAD: f64 = 0.3;

/// Samples of `bursts` of symbols sent back to back in `profile`, with
/// silence either side.
pub fn line(profile: Profile, bursts: &[Vec<dsp::Complex>]) -> Vec<f32> {
    let mut modulator = Modulator::new(profile, -12.0);
    let mut samples = Vec::new();
    modulator.fill((LEAD * FS) as usize, &mut samples);
    for symbols in bursts {
        modulator.push(symbols);
    }
    samples.extend(modulator.drain());
    samples.extend(std::iter::repeat_n(0.0, (LEAD * FS) as usize));
    samples
}

/// Everything a receiver listening for both profiles makes of `samples`,
/// printed as it goes.
pub fn receive(samples: &[f32]) -> Vec<Event> {
    let mut rx = Receiver::new(&Profile::ALL);
    rx.feed_all(samples);
    rx.finish();
    let events: Vec<Event> = std::iter::from_fn(|| rx.event()).collect();
    for e in &events {
        match e {
            Event::Heard { profile, header, snr_db, offset_hz, seconds } => println!(
                "  heard {} {:?} {} {} x{} at {seconds:.2} s, {snr_db:.1} dB, {offset_hz:+.2} Hz",
                profile.name(),
                header.kind,
                header.modulation.label(),
                header.rate.label(),
                header.codewords
            ),
            Event::Trained { snr_db } => println!("  trained {snr_db:.1} dB"),
            Event::BurstEnd(r) => println!(
                "  burst: {} ok, {} failed, {} slips, {:.1} dB, {:+.2} Hz, {:+.0} ppm{}",
                r.ok,
                r.failed,
                r.slips,
                r.snr_db,
                r.offset_hz,
                r.drift_ppm,
                if r.aborted { ", aborted" } else { "" }
            ),
            Event::Untrained { .. } | Event::Slip { .. } => println!("  {e:?}"),
            _ => {}
        }
    }
    events
}
