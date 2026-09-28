//! The headless modes: speech to a modem recording and back, a self test
//! through a simulated radio channel, the same through the live engine, and
//! what audio this machine has.

use std::path::Path;

use modem::channel::Channel;
use modem::profile::FS;
use modem::{Event, Modulator, Profile, Receiver, VOICE_MODES, VoiceMode, VoiceTx};
use voice::{Listener, Rate, Talker};

pub const USAGE: &str = "\
voicemodem                                   the window: talk and listen through a radio
voicemodem tx <speech.wav> <out.wav> [opts]  speech to the modem's audio, as a recording
voicemodem rx <in.wav> [speech.wav]          a recording of the modem back to speech
voicemodem selftest [opts]                   speech through a simulated radio channel
voicemodem loop [opts]                       the same in real time, through the live engine
voicemodem compare <a.wav> <b.wav>           how far the second recording strays from the first
voicemodem modes                             the voice modes
voicemodem devices                           the audio devices this machine has

tx options:
  --mode <name>        voice mode (default narrow-robust); see `modes`
  --text <text>        a short text sent alongside, a callsign most often
  --level <dBFS>       transmit level, rms (default -12)

selftest options:
  --mode <name>        voice mode (default narrow-robust)
  --seconds <s>        how much speech (default 20)
  --speech <file.wav>  real speech instead of the built-in synthetic voice
  --snr <dB>           Es/N0 of added noise
  --shift <Hz>         mistuning, as SSB passes it to the audio
  --doppler <Hz/s>     the mistuning moving, as a satellite's Doppler does
  --fade <dB>,<Hz>     slow fading: depth, and how often
  --ppm <ppm>          the far sound card's clock
  --save <dir>         write the modem audio and the speech that came back

loop options:
  --mode, --seconds, --save as for selftest
  --snr <dB>           the loopback's noise (default 20)
  --radio-in <dev> --radio-out <dev>
                       through two sound-card devices wired together instead,
                       a virtual cable most easily; see `devices`";

/// Options every mode might take, parsed from what follows the mode's name.
#[derive(Debug, Default)]
struct Opts {
    positional: Vec<String>,
    mode: Option<&'static VoiceMode>,
    text: Option<String>,
    level: Option<f64>,
    seconds: Option<f64>,
    speech: Option<String>,
    snr: Option<f64>,
    shift: Option<f64>,
    doppler: Option<f64>,
    fade: Option<(f64, f64)>,
    ppm: Option<f64>,
    save: Option<String>,
    radio_in: Option<String>,
    radio_out: Option<String>,
}

fn number(name: &str, value: &str) -> Result<f64, String> {
    value.trim().parse().map_err(|_| format!("{name} wants a number, not {value:?}"))
}

fn parse(args: &[String]) -> Result<Opts, String> {
    let mut opts = Opts::default();
    let mut rest = args.iter();
    while let Some(arg) = rest.next() {
        let mut value = |name: &str| rest.next().cloned().ok_or_else(|| format!("{name} needs a value"));
        match arg.as_str() {
            "--mode" => {
                let name = value("--mode")?;
                opts.mode = Some(VoiceMode::by_name(&name).ok_or_else(|| format!("no voice mode {name:?}; see `voicemodem modes`"))?);
            }
            "--text" => opts.text = Some(value("--text")?),
            "--level" => opts.level = Some(number("--level", &value("--level")?)?),
            "--seconds" => opts.seconds = Some(number("--seconds", &value("--seconds")?)?),
            "--speech" => opts.speech = Some(value("--speech")?),
            "--snr" => opts.snr = Some(number("--snr", &value("--snr")?)?),
            "--shift" => opts.shift = Some(number("--shift", &value("--shift")?)?),
            "--doppler" => opts.doppler = Some(number("--doppler", &value("--doppler")?)?),
            "--ppm" => opts.ppm = Some(number("--ppm", &value("--ppm")?)?),
            "--fade" => {
                let v = value("--fade")?;
                let (depth, rate) = v.split_once(',').ok_or("--fade wants <dB>,<Hz>")?;
                opts.fade = Some((number("--fade", depth)?, number("--fade", rate)?));
            }
            "--save" => opts.save = Some(value("--save")?),
            "--radio-in" => opts.radio_in = Some(value("--radio-in")?),
            "--radio-out" => opts.radio_out = Some(value("--radio-out")?),
            other if other.starts_with("--") => return Err(format!("unknown option {other}")),
            other => opts.positional.push(other.to_string()),
        }
    }
    Ok(opts)
}

/// Run a headless mode, if `args` name one. None if they name the window.
pub fn run(args: &[String]) -> Option<Result<(), String>> {
    let (first, rest) = args.split_first()?;
    let result = match first.as_str() {
        "tx" => parse(rest).and_then(|o| tx(&o)),
        "rx" => parse(rest).and_then(|o| rx(&o)),
        "selftest" | "self-test" => parse(rest).and_then(|o| selftest(&o)),
        "loop" => parse(rest).and_then(|o| live_loop(&o)),
        "compare" => parse(rest).and_then(|o| compare(&o)),
        "modes" => {
            modes();
            Ok(())
        }
        "devices" | "--devices" => {
            devices();
            Ok(())
        }
        "help" | "--help" | "-h" => {
            println!("{USAGE}");
            Ok(())
        }
        other => Err(format!("unknown mode {other:?}\n\n{USAGE}")),
    };
    Some(result)
}

fn modes() {
    println!(
        "{:<16} {:<13} {:<10} {:<13} {:>9} {:>8}  here",
        "mode", "profile", "modem", "codec", "latency", "frames"
    );
    let mut missing = None;
    for m in &VOICE_MODES {
        let here = voice::available(m.codec);
        println!(
            "{:<16} {:<13} {:<10} {:<13} {:>7.2} s {:>8}  {}",
            m.name,
            m.profile.label(),
            format!("{} {}", m.modulation.label(), m.rate.label()),
            m.codec.label(),
            m.codeword_seconds(),
            m.frames_per_codeword(),
            if here.is_ok() { "ready" } else { "needs weights" },
        );
        missing = missing.or(here.err());
    }
    println!("\nlatency is one codeword's air time; mouth to ear is about twice it.");
    if let Some(why) = missing {
        println!("{why}");
    }
}

fn devices() {
    println!("input devices:");
    for name in line::input_devices() {
        println!("  {name}");
    }
    println!("\noutput devices:");
    for name in line::output_devices() {
        println!("  {name}");
    }
}

/// A WAV's samples, mono, at `rate`.
fn read_at(path: &str, rate: f64) -> Result<Vec<f32>, String> {
    let wav = line::wav::read(path).map_err(|e| format!("{path}: {e}"))?;
    let mut out = Vec::new();
    Rate::new(f64::from(wav.sample_rate), rate).process(&wav.mono(), &mut out);
    Ok(out)
}

fn write(path: &Path, samples: &[f32], rate: u32) -> Result<(), String> {
    line::wav::write(path, samples, rate).map_err(|e| format!("{}: {e}", path.display()))
}

/// The modem's audio for `speech` (at `FS`) spoken in `mode`.
fn transmit(mode: &'static VoiceMode, speech: &[f32], text: &str, level: f64) -> Result<Vec<f32>, String> {
    let mut talker = Talker::new(mode, FS)?;
    let mut frames = Vec::new();
    talker.speak(speech, &mut frames);
    talker.finish(&mut frames);
    let mut tx = VoiceTx::new(mode, stream_id(), text);
    for frame in frames {
        tx.push_frame(frame);
    }
    tx.end();
    let mut modulator = Modulator::new(mode.profile, level);
    let mut samples = Vec::new();
    modulator.fill((0.25 * FS) as usize, &mut samples);
    while let Some(symbols) = tx.next_symbols() {
        modulator.push(&symbols);
    }
    samples.extend(modulator.drain());
    samples.extend(std::iter::repeat_n(0.0, (0.25 * FS) as usize));
    Ok(samples)
}

/// What a receiver makes of `line` (at `FS`): the speech, and a count of
/// codewords heard and lost.
struct Heard {
    speech: Vec<f32>,
    heard: usize,
    lost: usize,
    text: String,
}

fn receive(line: &[f32], verbose: bool) -> Heard {
    let mut rx = Receiver::new(&Profile::ALL);
    let mut listener: Option<Listener> = None;
    let mut out = Heard { speech: Vec::new(), heard: 0, lost: 0, text: String::new() };
    let on_event = |event: Event, out: &mut Heard, listener: &mut Option<Listener>| match event {
        Event::Heard { profile, header, snr_db, offset_hz, seconds } => {
            if verbose {
                let mode = VoiceMode::of(profile, &header).map_or("data", |m| m.name);
                println!("{seconds:7.2} s  heard {mode}, {snr_db:.1} dB, {offset_hz:+.1} Hz");
            }
        }
        Event::Voice { mode, codeword, .. } => {
            if listener.as_ref().is_none_or(|l| l.mode() != mode) {
                *listener = Listener::new(mode, FS).ok();
            }
            if let Some(l) = listener {
                l.hear(codeword.as_ref(), &mut out.speech);
                out.text = l.text();
            }
            if codeword.is_some() {
                out.heard += 1;
            } else {
                out.lost += 1;
            }
        }
        Event::BurstEnd(r) if verbose => println!(
            "           burst: {} ok, {} lost, {:.1} dB, {:+.1} Hz, {:+.0} ppm{}",
            r.ok,
            r.failed,
            r.snr_db,
            r.offset_hz,
            r.drift_ppm,
            if r.aborted { ", cut short" } else { "" }
        ),
        _ => {}
    };
    for &x in line {
        rx.feed(x);
        while let Some(e) = rx.event() {
            on_event(e, &mut out, &mut listener);
        }
    }
    rx.finish();
    while let Some(e) = rx.event() {
        on_event(e, &mut out, &mut listener);
    }
    out
}

fn tx(opts: &Opts) -> Result<(), String> {
    let [input, output] = opts.positional.as_slice() else {
        return Err("tx needs <speech.wav> <out.wav>".into());
    };
    let mode = opts.mode.unwrap_or(VoiceMode::robust(Profile::Narrow));
    let speech = read_at(input, FS)?;
    let samples = transmit(mode, &speech, opts.text.as_deref().unwrap_or(""), opts.level.unwrap_or(-12.0))?;
    write(Path::new(output), &samples, FS as u32)?;
    println!(
        "{}: {:.1} s of speech in {:.1} s of {} {} with {}",
        output,
        speech.len() as f64 / FS,
        samples.len() as f64 / FS,
        mode.modulation.label(),
        mode.rate.label(),
        mode.codec.label()
    );
    Ok(())
}

fn rx(opts: &Opts) -> Result<(), String> {
    let Some(input) = opts.positional.first() else {
        return Err("rx needs <in.wav>".into());
    };
    let line = read_at(input, FS)?;
    let heard = receive(&line, true);
    println!("{} codewords heard, {} lost; text {:?}", heard.heard, heard.lost, heard.text);
    if let Some(output) = opts.positional.get(1) {
        write(Path::new(output), &heard.speech, FS as u32)?;
        println!("{output}: {:.1} s of speech", heard.speech.len() as f64 / FS);
    }
    Ok(())
}

fn selftest(opts: &Opts) -> Result<(), String> {
    let mode = opts.mode.unwrap_or(VoiceMode::robust(Profile::Narrow));
    let seconds = opts.seconds.unwrap_or(20.0);
    let speech = match &opts.speech {
        Some(path) => read_at(path, FS)?,
        None => voice::synthetic_speech(seconds, FS),
    };
    let mut channel = Channel::clean().profile(mode.profile);
    if let Some(snr) = opts.snr {
        channel = channel.noise(snr);
    }
    if let Some(shift) = opts.shift {
        channel = channel.shift(shift);
    }
    if let Some(rate) = opts.doppler {
        channel = channel.doppler(rate);
    }
    if let Some((depth, rate)) = opts.fade {
        channel = channel.fading(depth, rate);
    }
    if let Some(ppm) = opts.ppm {
        channel = channel.clock(ppm);
    }
    println!(
        "{}: {} {} with {}, {:.0} to {:.0} Hz",
        mode.name,
        mode.modulation.label(),
        mode.rate.label(),
        mode.codec.label(),
        mode.profile.band().0,
        mode.profile.band().1
    );
    let sent = transmit(mode, &speech, "SELFTEST ", -12.0)?;
    let line = channel.apply(&sent);
    let started = std::time::Instant::now();
    let heard = receive(&line, true);
    let took = started.elapsed().as_secs_f64();
    let total = heard.heard + heard.lost;
    println!(
        "{} of {total} codewords heard ({:.1}%), text {:?}; received {:.1} s of line in {took:.2} s ({:.0}x real time)",
        heard.heard,
        100.0 * heard.heard as f64 / total.max(1) as f64,
        heard.text,
        line.len() as f64 / FS,
        line.len() as f64 / FS / took.max(1e-9)
    );
    if let Some(dir) = &opts.save {
        let dir = Path::new(dir);
        std::fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
        write(&dir.join("speech-in.wav"), &speech, FS as u32)?;
        write(&dir.join("line.wav"), &line, FS as u32)?;
        write(&dir.join("speech-out.wav"), &heard.speech, FS as u32)?;
        println!("wrote speech-in.wav, line.wav and speech-out.wav to {}", dir.display());
    }
    if total == 0 || heard.heard == 0 {
        return Err("nothing was heard".into());
    }
    Ok(())
}

/// The self test in real time: the live engine, speaking synthetic speech
/// into a loopback or a pair of sound-card devices wired together, and
/// hearing it back.
fn live_loop(opts: &Opts) -> Result<(), String> {
    use std::time::{Duration, Instant};

    use crate::engine::Command;

    let mode = opts.mode.unwrap_or(VoiceMode::robust(Profile::Narrow));
    let seconds = opts.seconds.unwrap_or(15.0);
    let radio = match (&opts.radio_in, &opts.radio_out) {
        (Some(i), Some(o)) => Some((i.clone(), o.clone())),
        (None, None) => None,
        _ => return Err("--radio-in and --radio-out go together; see `voicemodem devices`".into()),
    };
    let (commands, status) = crate::engine::spawn(mode);
    let send = |c: Command| commands.send(c).map_err(|_| "the engine stopped".to_string());
    send(Command::Open { radio, loopback_snr: opts.snr.unwrap_or(20.0), operator: None })?;
    send(Command::Mode(mode))?;
    send(Command::Text("LOOP ".into()))?;
    // A cable loop hears its own transmission, which is the point of it.
    send(Command::FullDuplex(true))?;
    send(Command::Record(true))?;
    // Let the sound cards settle before speaking.
    std::thread::sleep(Duration::from_millis(600));
    let speech = voice::synthetic_speech(seconds, FS);
    println!("{}: speaking {seconds:.0} s in real time", mode.name);
    send(Command::Speak(speech.clone()))?;
    send(Command::Ptt(true))?;
    std::thread::sleep(Duration::from_secs_f64(seconds));
    send(Command::Ptt(false))?;
    let started = Instant::now();
    loop {
        std::thread::sleep(Duration::from_millis(100));
        let transmitting = status.lock().map(|s| s.transmitting).unwrap_or(false);
        if !transmitting || started.elapsed() > Duration::from_secs(20) {
            break;
        }
    }
    // The last codeword's speech, played out.
    std::thread::sleep(Duration::from_secs_f64(2.0 * mode.codeword_seconds() + 0.5));
    let (reply, recording) = std::sync::mpsc::channel();
    send(Command::TakeRecording(reply))?;
    let heard = recording.recv_timeout(Duration::from_secs(2)).unwrap_or_default();
    let view = status.lock().map(|s| s.clone()).map_err(|_| "the engine's status is poisoned")?;
    let _ = commands.send(Command::Quit);
    for line in &view.log {
        println!("  {line}");
    }
    let total = view.heard + view.lost;
    println!(
        "{} of {total} codewords heard, text {:?}; {:.1} s of speech sent, {:.1} s played out",
        view.heard,
        view.text,
        speech.len() as f64 / FS,
        heard.len() as f64 / FS
    );
    if let Some(dir) = &opts.save {
        let dir = Path::new(dir);
        std::fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
        write(&dir.join("loop-speech-in.wav"), &speech, FS as u32)?;
        write(&dir.join("loop-speech-out.wav"), &heard, FS as u32)?;
        println!("wrote loop-speech-in.wav and loop-speech-out.wav to {}", dir.display());
    }
    if view.heard == 0 {
        return Err("nothing was heard".into());
    }
    Ok(())
}

/// Two recordings of the same speech, the second against the first.
fn compare(opts: &Opts) -> Result<(), String> {
    let [a, b] = opts.positional.as_slice() else {
        return Err("compare needs <a.wav> <b.wav>".into());
    };
    let c = crate::compare::compare(&read_at(a, FS)?, &read_at(b, FS)?);
    println!(
        "{b} against {a}: {:+.0} ms late, {:+.1} dB, log-spectral distance {:.2} dB over {} frames, envelope correlation {:.3}",
        c.delay_ms, c.gain_db, c.lsd_db, c.frames, c.envelope
    );
    Ok(())
}

/// A transmission's number: anything that differs from the last one.
pub fn stream_id() -> u16 {
    let nanos = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.subsec_nanos());
    (nanos ^ (nanos >> 16)) as u16
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn options_parse() {
        let args: Vec<String> = ["a.wav", "--mode", "wide-robust", "--fade", "12,0.5", "--snr", "6"].map(String::from).to_vec();
        let o = parse(&args).unwrap();
        assert_eq!(o.positional, vec!["a.wav"]);
        assert_eq!(o.mode.map(|m| m.name), Some("wide-robust"));
        assert_eq!(o.fade, Some((12.0, 0.5)));
        assert_eq!(o.snr, Some(6.0));
        assert!(parse(&["--mode".to_string(), "nope".to_string()]).is_err());
    }

    #[test]
    fn speech_goes_through_the_modem_and_comes_back() {
        let mode = VoiceMode::robust(Profile::Wide);
        let speech = voice::synthetic_speech(4.0, FS);
        let line = transmit(mode, &speech, "TEST", -12.0).unwrap();
        let heard = receive(&line, false);
        assert!(heard.heard > 0 && heard.lost == 0);
        assert!(heard.text.contains("TEST"));
        // All the speech, give or take the codec's part-frame at the end and
        // the two rate conversions' kernels either side of it.
        let frame = (mode.codec.frame_seconds() * FS) as usize + 64;
        assert!(heard.speech.len() + frame >= speech.len(), "{} of {}", heard.speech.len(), speech.len());
    }
}
