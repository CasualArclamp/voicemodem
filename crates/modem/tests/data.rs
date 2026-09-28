//! Text and files through simulated lines, in both profiles.

mod common;

use common::{LEAD, line, receive};
use modem::channel::Channel;
use modem::profile::FS;
use modem::{Content, Delivery, Event, Modulation, Outgoing, Profile, Rate};

fn delivered(events: &[Event]) -> Vec<Delivery> {
    events
        .iter()
        .filter_map(|e| match e {
            Event::Delivered(Ok(d)) => Some(d.clone()),
            _ => None,
        })
        .collect()
}

fn bursts_of(out: &Outgoing) -> Vec<Vec<dsp::Complex>> {
    (0..out.bursts()).map(|b| out.burst(b).1).collect()
}

/// Send `content` in `profile`, `modulation` and `rate` through `channel`,
/// and whether it came out whole.
fn crosses(content: &Content, profile: Profile, modulation: Modulation, rate: Rate, burst: usize, channel: Channel) -> bool {
    let out = Outgoing::new(content, 0x1234, modulation, rate).unwrap().with_burst_length(burst);
    let samples = channel.profile(profile).apply(&line(profile, &bursts_of(&out)));
    println!("{} {} {}: {} blocks in {} bursts", profile.name(), modulation.label(), rate.label(), out.blocks(), out.bursts());
    delivered(&receive(&samples)).iter().any(|d| d.content == *content)
}

fn text() -> Content {
    Content::Text("VK2XYZ testing, three phases. The quick brown fox jumps over the lazy dog.".into())
}

fn file(bytes: usize) -> Content {
    Content::File { name: "test.bin".into(), data: (0..bytes).map(|i| (i * 7 + i / 251) as u8).collect() }
}

#[test]
fn every_mode_carries_a_message_over_a_clean_line() {
    for profile in Profile::ALL {
        for modulation in Modulation::ALL {
            for rate in Rate::ALL {
                assert!(
                    crosses(&text(), profile, modulation, rate, 8, Channel::clean()),
                    "{} {} {}",
                    profile.name(),
                    modulation.label(),
                    rate.label()
                );
            }
        }
    }
}

#[test]
fn a_file_crosses_in_several_bursts() {
    // 2000 bytes of QPSK 1/2 is twenty blocks: five bursts of four.
    for profile in Profile::ALL {
        assert!(crosses(&file(2000), profile, Modulation::Qpsk, Rate::Half, 4, Channel::clean().noise(20.0)));
    }
}

#[test]
fn a_far_clock_off_by_hundreds_of_ppm_is_followed() {
    for ppm in [-200.0, 150.0] {
        let channel = Channel::clean().clock(ppm).noise(25.0);
        assert!(crosses(&file(3000), Profile::Wide, Modulation::Psk8, Rate::ThreeQuarters, 16, channel), "{ppm} ppm");
    }
}

#[test]
fn a_telephone_channel_through_mulaw() {
    // Fourth-order band edges and their group delay, G.711 at 8 kHz, a far
    // sound card 100 ppm fast and a 3 Hz carrier offset: a radio linked over
    // a VoIP path.
    let channel = Channel::clean().telephone().mulaw().clock(100.0).shift(3.0);
    assert!(crosses(&file(2500), Profile::Wide, Modulation::Psk8, Rate::ThreeQuarters, 16, channel));
}

#[test]
fn jitter_buffer_slips_either_way_are_survived() {
    // 20 ms played twice, and 20 ms dropped, a second into the payload: 48
    // symbols wide, 32 narrow.
    for by in [320, -320] {
        let at = ((LEAD + 1.0) * FS) as usize;
        for (profile, modulation, rate) in
            [(Profile::Wide, Modulation::Qpsk, Rate::Half), (Profile::Narrow, Modulation::Psk8, Rate::TwoThirds)]
        {
            let channel = Channel::clean().slip(at, by).noise(22.0);
            assert!(
                crosses(&file(2000), profile, modulation, rate, 16, channel),
                "{} {} {} slip {by}",
                profile.name(),
                modulation.label(),
                rate.label()
            );
        }
    }
}

#[test]
fn quiet_and_loud_lines_both_work() {
    for gain in [-36.0, 8.0] {
        let channel = Channel::clean().gain(gain).noise(25.0);
        assert!(crosses(&text(), Profile::Narrow, Modulation::Qpsk, Rate::TwoThirds, 8, channel), "{gain} dB");
    }
}

#[test]
fn two_transfers_in_different_profiles_both_arrive() {
    let first = Outgoing::new(&text(), 1, Modulation::Bpsk, Rate::Half).unwrap();
    let second = Outgoing::new(&file(700), 2, Modulation::Psk8, Rate::Half).unwrap();
    let mut samples = line(Profile::Narrow, &bursts_of(&first));
    samples.extend(line(Profile::Wide, &bursts_of(&second)));
    let got = delivered(&receive(&Channel::clean().noise(20.0).apply(&samples)));
    assert_eq!(got.len(), 2, "{got:?}");
    assert_eq!(got[0].content, text());
    assert_eq!(got[1].content, file(700));
}

#[test]
fn noise_alone_delivers_nothing() {
    let noise = Channel::clean().noise(0.0).apply(&vec![0.05; 16_000 * 6]);
    let events = receive(&noise);
    assert!(events.iter().all(|e| !matches!(e, Event::Delivered(_) | Event::Trained { .. })), "{events:?}");
}
