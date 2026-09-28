//! The codecs, each on a thread of its own.
//!
//! The engine's loop has ten milliseconds a turn and a sound card waiting on
//! it at each end. Codec 2 fits in that many times over; EnCodec does not --
//! a codeword of its frames takes the model a sixth of a second to decode on
//! an ordinary CPU, and a chunk of speech some forty milliseconds to encode.
//! Run in the loop, that starved the radio's output and dropped its input,
//! which on the air is a transmission with holes in it. So the loop only
//! hands work over and picks up what is finished; the waiting happens here.

use std::sync::mpsc::{Receiver, Sender, channel};

use modem::profile::FS;
use modem::{Codec, VoiceCodeword, VoiceMode};
use voice::{Listener, Talker};

enum TalkJob {
    Start(&'static VoiceMode),
    Speech(Vec<f32>),
    Finish,
}

/// What the talker's thread hands back.
#[derive(Debug)]
pub enum Talked {
    /// Codec frames, in order.
    Frames(Vec<Vec<u8>>),
    /// Everything spoken before [`TalkerThread::finish`] has been handed back.
    Finished,
    Failed(String),
}

/// Speech in, codec frames out, on a thread of its own.
#[derive(Debug)]
pub struct TalkerThread {
    jobs: Sender<TalkJob>,
    done: Receiver<Talked>,
}

impl TalkerThread {
    pub fn spawn() -> Self {
        let (jobs, inbox) = channel::<TalkJob>();
        let (outbox, done) = channel();
        std::thread::Builder::new()
            .name("talker".into())
            .spawn(move || {
                let mut talker: Option<Talker> = None;
                while let Ok(job) = inbox.recv() {
                    let mut frames = Vec::new();
                    let finishing = matches!(job, TalkJob::Finish);
                    match job {
                        TalkJob::Start(mode) => match Talker::new(mode, FS) {
                            Ok(t) => talker = Some(t),
                            Err(e) => {
                                talker = None;
                                let _ = outbox.send(Talked::Failed(e));
                            }
                        },
                        TalkJob::Speech(speech) => {
                            if let Some(t) = &mut talker {
                                t.speak(&speech, &mut frames);
                            }
                        }
                        TalkJob::Finish => {
                            if let Some(t) = &mut talker {
                                t.finish(&mut frames);
                            }
                            talker = None;
                        }
                    }
                    if !frames.is_empty() && outbox.send(Talked::Frames(frames)).is_err() {
                        return;
                    }
                    if finishing && outbox.send(Talked::Finished).is_err() {
                        return;
                    }
                }
            })
            .expect("the talker thread starts");
        Self { jobs, done }
    }

    /// A new transmission in `mode`: a fresh codec.
    pub fn start(&self, mode: &'static VoiceMode) {
        let _ = self.jobs.send(TalkJob::Start(mode));
    }

    /// Speech at the modem's rate.
    pub fn speak(&self, speech: Vec<f32>) {
        if !speech.is_empty() {
            let _ = self.jobs.send(TalkJob::Speech(speech));
        }
    }

    /// The speaker has let go: hand back whatever the codec still holds,
    /// then [`Talked::Finished`].
    pub fn finish(&self) {
        let _ = self.jobs.send(TalkJob::Finish);
    }

    /// What the thread has finished since last asked.
    pub fn done(&self) -> Vec<Talked> {
        self.done.try_iter().collect()
    }
}

enum HearJob {
    Codeword(&'static VoiceMode, Option<VoiceCodeword>),
    Warm(Codec),
}

/// What the listener's thread hands back.
#[derive(Debug)]
pub enum Heard {
    /// Speech at the modem's rate, and the transmission's text so far.
    Speech(Vec<f32>, String),
    Failed(String),
}

/// Codewords in, speech out, on a thread of its own.
#[derive(Debug)]
pub struct ListenerThread {
    jobs: Sender<HearJob>,
    done: Receiver<Heard>,
}

impl ListenerThread {
    pub fn spawn() -> Self {
        let (jobs, inbox) = channel::<HearJob>();
        let (outbox, done) = channel();
        std::thread::Builder::new()
            .name("listener".into())
            .spawn(move || {
                let mut listener: Option<Listener> = None;
                // A mode that would not open is said once, not every codeword.
                let mut refused: Option<&'static VoiceMode> = None;
                while let Ok(job) = inbox.recv() {
                    match job {
                        HearJob::Warm(codec) => {
                            // Opened for its side effect: a neural codec's
                            // weights loaded now, off the audio's path.
                            if let Err(e) = voice::open(codec) {
                                let _ = outbox.send(Heard::Failed(format!("{} cannot be used: {e}", codec.label())));
                            }
                        }
                        HearJob::Codeword(mode, codeword) => {
                            if listener.as_ref().is_none_or(|l| l.mode() != mode) {
                                listener = None;
                                match Listener::new(mode, FS) {
                                    Ok(l) => {
                                        listener = Some(l);
                                        refused = None;
                                    }
                                    Err(e) => {
                                        if refused != Some(mode) {
                                            refused = Some(mode);
                                            let _ = outbox.send(Heard::Failed(format!("cannot play {}: {e}", mode.name)));
                                        }
                                        continue;
                                    }
                                }
                            }
                            if let Some(l) = &mut listener {
                                let mut speech = Vec::new();
                                l.hear(codeword.as_ref(), &mut speech);
                                if outbox.send(Heard::Speech(speech, l.text())).is_err() {
                                    return;
                                }
                            }
                        }
                    }
                }
            })
            .expect("the listener thread starts");
        Self { jobs, done }
    }

    /// A codeword as the receiver had it, None if it did not decode.
    pub fn hear(&self, mode: &'static VoiceMode, codeword: Option<VoiceCodeword>) {
        let _ = self.jobs.send(HearJob::Codeword(mode, codeword));
    }

    /// Open `codec` now, so that its weights are loaded before they are
    /// needed; any failure comes back as [`Heard::Failed`].
    pub fn warm(&self, codec: Codec) {
        let _ = self.jobs.send(HearJob::Warm(codec));
    }

    /// What the thread has finished since last asked.
    pub fn done(&self) -> Vec<Heard> {
        self.done.try_iter().collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use modem::Profile;
    use std::time::{Duration, Instant};

    /// Everything a thread hands back, until `enough` says it is, or five
    /// seconds have gone.
    fn collect<T>(poll: impl Fn() -> Vec<T>, enough: impl Fn(&[T]) -> bool) -> Vec<T> {
        let started = Instant::now();
        let mut all = Vec::new();
        while !enough(&all) && started.elapsed() < Duration::from_secs(5) {
            all.extend(poll());
            std::thread::sleep(Duration::from_millis(5));
        }
        all
    }

    #[test]
    fn the_talker_hands_back_every_frame_and_then_says_it_has() {
        let mode = VoiceMode::robust(Profile::Narrow);
        let talker = TalkerThread::spawn();
        talker.start(mode);
        // Two seconds of speech, and a little more that only finishing sends.
        talker.speak(voice::synthetic_speech(2.01, FS));
        talker.finish();
        let done = collect(|| talker.done(), |all| all.iter().any(|d| matches!(d, Talked::Finished)));
        let frames: usize = done.iter().map(|d| if let Talked::Frames(f) = d { f.len() } else { 0 }).sum();
        assert!(matches!(done.last(), Some(Talked::Finished)), "{done:?}");
        assert!((50..=52).contains(&frames), "{frames} frames for 2 s at 40 ms");
    }

    #[test]
    fn the_listener_hands_back_speech_and_text() {
        let mode = VoiceMode::robust(Profile::Narrow);
        let listener = ListenerThread::spawn();
        let frame = vec![0u8; mode.codec.bits()];
        let codeword = VoiceCodeword { seq: 0, frames: vec![frame; 10], end: false, text: b'K' };
        listener.hear(mode, Some(codeword));
        listener.hear(mode, None);
        let done = collect(|| listener.done(), |all| all.len() >= 2);
        let speech: Vec<usize> = done.iter().map(|d| if let Heard::Speech(s, _) = d { s.len() } else { 0 }).collect();
        // Ten 40 ms frames, then a lost codeword's air time of silence;
        // the first a little short, by the rate conversion's kernel.
        assert_eq!(speech.len(), 2, "{done:?}");
        assert!(speech[0].abs_diff((0.4 * FS) as usize) < 64, "{} samples", speech[0]);
        assert!(matches!(&done[0], Heard::Speech(_, text) if text == "K"));
    }
}
