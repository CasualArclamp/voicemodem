//! Voice transmissions through simulated radio channels: every mode, and
//! what SSB, FM and a satellite pass do to them.
//!
//! The frames are random bits rather than speech: the modem's job is to
//! deliver every codec frame exactly, and a frame that arrives is checked
//! bit for bit against the one sent.

mod common;

use common::{LEAD, line, receive};
use modem::channel::{Channel, Random};
use modem::profile::FS;
use modem::{Event, Profile, VOICE_MODES, VoiceMode, VoiceTx};

/// `n` frames of random bits for `mode`'s codec.
fn frames(mode: &VoiceMode, n: usize, seed: u64) -> Vec<Vec<u8>> {
    let mut random = Random::new(seed);
    (0..n).map(|_| (0..mode.codec.bits()).map(|_| (random.next_u64() >> 63) as u8).collect()).collect()
}

/// The symbols of one transmission of `sent`, spoken all at once and then
/// let go.
fn transmission(mode: &'static VoiceMode, sent: &[Vec<u8>]) -> Vec<dsp::Complex> {
    let mut tx = VoiceTx::new(mode, 0x77, "VK2TEST ");
    for frame in sent {
        tx.push_frame(frame.clone());
    }
    tx.end();
    std::iter::from_fn(|| tx.next_symbols()).flatten().collect()
}

#[derive(Debug, Default)]
struct Heard {
    /// Frames of every codeword that decoded, in order.
    frames: Vec<Vec<u8>>,
    ok: usize,
    lost: usize,
    text: String,
    ended: bool,
}

fn hear(events: &[Event]) -> Heard {
    let mut heard = Heard::default();
    for e in events {
        if let Event::Voice { codeword, .. } = e {
            match codeword {
                Some(c) => {
                    heard.ok += 1;
                    heard.frames.extend(c.frames.iter().cloned());
                    if c.text != 0 {
                        heard.text.push(char::from(c.text));
                    }
                    heard.ended |= c.end;
                }
                None => heard.lost += 1,
            }
        }
    }
    heard
}

/// `seconds` of speech in `mode` through `channel`: what was heard.
fn speak(mode: &'static VoiceMode, seconds: f64, channel: Channel) -> (Vec<Vec<u8>>, Heard) {
    let sent = frames(mode, (seconds / mode.codec.frame_seconds()) as usize, 7);
    let samples = channel.profile(mode.profile).apply(&line(mode.profile, &[transmission(mode, &sent)]));
    println!("{}: {} frames", mode.name, sent.len());
    (sent, hear(&receive(&samples)))
}

#[test]
fn every_mode_delivers_every_frame_over_a_clean_channel() {
    for mode in &VOICE_MODES {
        let (sent, heard) = speak(mode, 12.0, Channel::clean());
        assert_eq!(heard.lost, 0, "{}", mode.name);
        assert!(heard.frames == sent, "{}: {} frames of {} came back", mode.name, heard.frames.len(), sent.len());
        assert!(heard.ended, "{}: the end was not heard", mode.name);
        assert!(heard.text.contains("VK2TEST"), "{}: text {:?}", mode.name, heard.text);
    }
}

#[test]
fn the_robust_modes_hold_near_their_threshold() {
    // Es/N0 at the profile's symbol rate. Wide's robust mode is BPSK 2/3
    // and narrow's QPSK 1/2; both should lose almost nothing here.
    for (profile, snr) in [(Profile::Narrow, 5.0), (Profile::Wide, 4.0)] {
        let mode = VoiceMode::robust(profile);
        let (_, heard) = speak(mode, 20.0, Channel::clean().noise(snr).seed(3));
        let total = heard.ok + heard.lost;
        assert!(heard.lost * 20 <= total, "{} at {snr} dB: {} of {total} codewords lost", mode.name, heard.lost);
    }
}

#[test]
fn each_mode_holds_at_its_working_signal_to_noise() {
    let working = |mode: &VoiceMode| match (mode.modulation.bits(), mode.rate.label()) {
        (1, _) => 6.0,
        (2, "1/2") => 7.0,
        (2, "2/3") => 9.0,
        (2, _) => 10.0,
        (_, "2/3") => 14.0,
        _ => 16.0,
    };
    for mode in &VOICE_MODES {
        let snr = working(mode);
        let (_, heard) = speak(mode, 10.0, Channel::clean().noise(snr).seed(11));
        assert_eq!(heard.lost, 0, "{} at {snr} dB", mode.name);
    }
}

#[test]
fn a_mistuned_ssb_receiver_and_a_satellites_doppler_are_followed() {
    // Up to 400 Hz off, still moving at 8 Hz a second, as the residual of a
    // LEO pass after tracking software has corrected most of it.
    let mode = VoiceMode::robust(Profile::Narrow);
    for (shift, rate) in [(-400.0, 8.0), (230.0, -8.0), (60.0, 0.0)] {
        let (sent, heard) = speak(mode, 20.0, Channel::clean().shift(shift).doppler(rate).noise(12.0));
        assert_eq!(heard.lost, 0, "{shift} Hz moving {rate} Hz/s");
        assert_eq!(heard.frames, sent);
    }
}

#[test]
fn slow_deep_fading_costs_only_what_is_faded() {
    // A spinning satellite: down 12 dB and back twice a second.
    for profile in Profile::ALL {
        let mode = VoiceMode::robust(profile);
        let (_, heard) = speak(mode, 20.0, Channel::clean().fading(12.0, 0.5).noise(16.0).seed(5));
        let total = heard.ok + heard.lost;
        assert!(heard.lost * 10 <= total, "{}: {} of {total} codewords lost", mode.name, heard.lost);
    }
}

#[test]
fn a_far_sound_card_clock_is_followed() {
    for ppm in [-150.0, 150.0] {
        let mode = VoiceMode::by_name("narrow-high").unwrap();
        let (sent, heard) = speak(mode, 15.0, Channel::clean().clock(ppm).noise(22.0));
        assert_eq!(heard.frames, sent, "{ppm} ppm");
    }
}

#[test]
fn the_display_keeps_its_points_from_one_burst_to_the_next() {
    // Two transmissions: QPSK over several bursts, then 8PSK.
    let qpsk = VoiceMode::robust(Profile::Narrow);
    let psk8 = VoiceMode::by_name("narrow-high").unwrap();
    let mut bursts = vec![transmission(qpsk, &frames(qpsk, (14.0 / qpsk.codec.frame_seconds()) as usize, 1))];
    bursts.push(transmission(psk8, &frames(psk8, (3.0 / psk8.codec.frame_seconds()) as usize, 2)));
    let samples = line(Profile::Narrow, &bursts);
    let mut rx = modem::Receiver::new(&Profile::ALL);
    let mut heard_with_points = Vec::new();
    for &x in &samples {
        rx.feed(x);
        while let Some(e) = rx.event() {
            if let Event::Heard { header, .. } = e {
                heard_with_points.push((header.modulation, rx.points().len()));
            }
        }
    }
    // Every preamble after the first finds the last burst's points still there.
    assert!(heard_with_points.len() >= 3, "{heard_with_points:?}");
    for (modulation, points) in &heard_with_points[1..] {
        assert!(*points > 0, "a {} preamble found the display empty: {heard_with_points:?}", modulation.label());
    }
    // And a new modulation starts it afresh, with only its own points.
    assert_eq!(rx.modulation(), Some(modem::Modulation::Psk8));
    let turned: Vec<f64> = rx.points().iter().map(|z| (z.arg() / (std::f64::consts::TAU / 8.0)).round()).collect();
    assert!(turned.iter().any(|k| k.rem_euclid(2.0) == 1.0), "no 8PSK points among {} shown", turned.len());
}

#[test]
fn a_receiver_that_tunes_in_late_joins_at_the_next_burst() {
    let mode = VoiceMode::robust(Profile::Wide);
    let sent = frames(mode, (30.0 / mode.codec.frame_seconds()) as usize, 9);
    let samples = line(mode.profile, &[transmission(mode, &sent)]);
    // Everything before three seconds in is never heard.
    let late = &samples[((LEAD + 3.0) * FS) as usize..];
    let heard = hear(&receive(late));
    assert!(heard.ok > 0 && heard.frames.len() < sent.len());
    // What was heard is the end of what was sent, exactly.
    assert_eq!(heard.frames[..], sent[sent.len() - heard.frames.len()..]);
}
