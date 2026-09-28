# voicemodem

Digital voice through a radio's voice channel — SSB or FM, satellite or
terrestrial. Speech goes through Codec 2 or EnCodec, then out as single-carrier
BPSK, QPSK or 8PSK. Every burst opens with a DBPSK preamble carrying
everything the receiver needs: where the symbols are, how far the carrier is
off, and what follows. One standalone `voicemodem.exe`, in Rust, with no C and
no installer.

The receiver is [BinModem](https://github.com/CasualArclamp/BinModem)'s,
from its live-proven QAM core. `crates/dsp` is BinModem's DSP crate, copied
with two small additions (see its `lib.rs`), so fixes can move between the two
projects by diff.

## Quick start

1. Run `voicemodem.exe`.
2. Tick **loopback, no radio**, choose a **mic** and **speaker**, and press
   **Open**.
3. Hold **PUSH TO TALK** (or Space) and speak. You hear yourself back through
   the whole modem, at whatever SNR the slider sets. This is the quickest way
   to hear what each mode and codec does before going on the air.

`voicemodem demo` does the same with no microphone. It opens the loopback
and talks to itself with synthetic speech for ten seconds of every fifteen,
so the scopes have something to show.

Click the constellation to draw it large. It is drawn the way BinModem draws
one, and it builds up across bursts, starting again only when the modulation
changes.

On the air, choose the rig's audio interface as **radio in** and **radio out**
and untick loopback. Key the rig with VOX or by hand. PTT through a serial
line or CAT is not built in yet.

## Modes

Two profiles are chosen per transmission; the receiver listens for both at
once and follows whichever it hears.

| profile | symbol rate | band | for |
|---|---|---|---|
| Narrow (SSB) | 1600 baud on 1500 Hz | 540–2460 Hz | HF, linear-transponder satellites |
| Wide (FM) | 2400 baud on 1800 Hz | 360–3240 Hz | repeaters, FM satellites, flat data ports |

| mode | modem | codec | every codeword at | one codeword |
|---|---|---|---|---|
| narrow-robust | QPSK 1/2 | Codec2 1200 | 5 dB | 0.64 s |
| narrow-standard | QPSK 2/3 | Codec2 1600 | 7 dB | 0.64 s |
| narrow-high | 8PSK 2/3 | Codec2 2400 | 12 dB | 0.64 s |
| narrow-neural | QPSK 2/3 | EnCodec 1.5k | 7 dB | 0.64 s |
| wide-robust | BPSK 2/3 | Codec2 1200 | 4 dB (94% at 2 dB) | 0.43 s |
| wide-standard | QPSK 1/2 | Codec2 1600 | 6 dB | 0.43 s |
| wide-high | QPSK 3/4 | Codec2 2400 | 7 dB | 0.43 s |
| wide-best | 8PSK 2/3 | Codec2 3200 | 11 dB | 0.43 s |
| wide-neural | QPSK 1/2 | EnCodec 1.5k | 6 dB | 0.43 s |
| wide-neural-hq | 8PSK 3/4 | EnCodec 3k | — | 0.43 s |

"Every codeword at" is the Es/N0 at which a 30 s transmission lost no
codewords, measured with `voicemodem selftest`. The neural rows reuse the
figure of their modem, since the codec does not change the modem. A lost
codeword is played as silence.

Mouth-to-ear delay, measured through the live engine with `voicemodem loop`,
is about 2.4 s narrow and 1.7 s wide with Codec 2, and 0.1 s more with
EnCodec. Most of it is the price of long, well-interleaved codewords: a
codeword is sent only once its speech has been spoken, and played only once
it has all arrived. The receiver also holds back enough speech to ride out
the next burst's preamble.

Narrow has no BPSK voice mode. At 1600 baud, BPSK nets at most 1050 bit/s
once the pilots are paid for, and Codec2's lowest rate in Rust is 1200. The
preamble is BPSK in every mode.

Also measured, by the tests in `crates/modem/tests/voice.rs`:

- SSB mistuning of up to ±400 Hz, still drifting at ±8 Hz/s (a LEO pass's
  residual Doppler after tracking software), with no codeword lost.
- 12 dB spin fading twice a second, with under 10% lost.
- A far sound card's clock ±150 ppm out.
- A receiver tuning in mid-transmission, which joins at the next burst.

## Setting up a radio

- **SSB.** Use USB and the rig's data mode if it has one, with a filter of at
  least 2.4 kHz. Keep the ALC at zero: PSK needs a linear chain, so set the
  level slider and the rig's input gain so the ALC never moves. On receive,
  turn the noise blanker off and the AGC to slow.
- **FM.** Use the flat data port if the rig has one. The mic and speaker path
  works too; the equaliser takes out the pre-emphasis tilt.
- **Satellites.** On a linear transponder, let Doppler software correct the
  frequency. The modem follows what is left: about ±450 Hz of offset, moving
  several hertz a second. An FM satellite needs nothing. Tick **decode while
  transmitting** to hear your own downlink.
- **Text.** The field under the mode is sent one character per codeword, over
  and over. Put your callsign there.

## EnCodec

The neural modes use Meta's [EnCodec](https://github.com/facebookresearch/encodec)
24 kHz model, run by [candle](https://github.com/huggingface/candle) in pure
Rust on the CPU. It runs at about five times real time each way.

The weights are **not** part of this program. They are Meta's, licensed
CC-BY-NC 4.0, which is fine for amateur radio (non-commercial by definition)
but not redistributable under the GPL. To use the neural modes:

1. Download `model.safetensors` (about 93 MB) from
   <https://huggingface.co/facebook/encodec_24khz>.
2. Rename it `encodec_24khz.safetensors`.
3. Put it next to `voicemodem.exe` or in `%APPDATA%\voicemodem\`, or point
   the `VOICEMODEM_ENCODEC` environment variable at it.

Choosing a neural mode without the file says where it looked.

Checked with Meta's weights on 15 s of synthesised English speech:

- Streamed in 200 ms chunks, the model's output matches the model run on the
  whole clip at once (r = 0.985, no offset).
- Its waveform correlates with the input at 0.85 (1.5k) and 0.91 (3k).

Measured through the whole modem at 20 dB, where every codeword arrived, with
`voicemodem compare`:

| mode | codec | log-spectral distance | envelope correlation |
|---|---|---|---|
| narrow-robust | Codec2 1200 | 9.0 dB | 0.976 |
| narrow-standard | Codec2 1600 | 8.8 dB | 0.893 |
| narrow-high | Codec2 2400 | 9.0 dB | 0.978 |
| wide-best | Codec2 3200 | 8.6 dB | 0.923 |
| narrow-neural | EnCodec 1.5k | 10.6 dB | 0.970 |
| wide-neural-hq | EnCodec 3k | 9.8 dB | 0.975 |

These are objective measures, not a listening test. Your ears decide which
codec sounds better.

## How it works

```text
 preamble: DBPSK, 300 symbols                      payload: coherent PSK
+---+-----------+-----------+--------------+     +-------+---------+-------+-----+-------+
| R | reversals | unique    | header       |     | pilot | data    | pilot | ... | pilot |
|   | 64        | word 63   | 172 (coded)  |     | 16    | 112     | 16    |     | 16    |
+---+-----------+-----------+--------------+     +-------+---------+-------+-----+-------+
```

- **Preamble.** Each symbol is the one before it, reversed or not. The
  receiver can read it before it knows the carrier's phase or frequency:
  multiplying each matched-filter output by the conjugate of the one a symbol
  earlier gives the differential bit, turned by the carrier offset. The
  63-chip unique word, correlated against those products, gives the symbol
  timing to a fraction of a sample and the carrier offset. The header carries
  data or voice, the modulation, code rate, codec, burst length and stream.
  It is K=7 coded and has a CRC-16.
- **Training.** Once the header checks, all 300 preamble symbols are known. A
  fresh copy of BinModem's QAM core is made for the burst, mixing down at the
  measured carrier. The line is played into it again from just before the
  preamble, and its 31-tap half-symbol equaliser is solved outright by least
  squares on those symbols. From then on it tracks timing, carrier and
  equaliser, and finds itself again after a fade from the stored samples.
- **Framing.** Sixteen known pilots every 128 symbols say whether the stream
  has moved by whole symbols (a network jitter buffer's 20 ms slip) and which
  way round the constellation is after a fade. They go out on a point of the
  payload's constellation and its opposite, so the receiver slices them with
  the data. Data between two pilots that disagree is erased rather than
  trusted.
- **Codewords.** Eight slots (896 data symbols) per codeword. Each has a
  whitener, a K=7 convolutional code punctured to 1/2, 2/3 or 3/4, an
  interleaver across the whole codeword, and Gray-mapped PSK: BPSK on ±1,
  QPSK on the diagonals (45°, 135°, 225°, 315°), and 8PSK every 45° from +1.
  A voice codeword carries whole codec frames, a frame count, a sequence
  number, an end flag, one text character and a CRC-16. Each codeword holds
  a little more speech than its own air time, which pays back the periodic
  preambles; that is how a late listener gets in.

## Command line

```text
voicemodem                                   the window
voicemodem tx <speech.wav> <out.wav> [opts]  speech to the modem's audio, as a recording
voicemodem rx <in.wav> [speech.wav]          a recording of the modem back to speech
voicemodem selftest [opts]                   speech through a simulated radio channel
voicemodem loop [opts]                       the same in real time, through the live engine
voicemodem compare <a.wav> <b.wav>           how far the second recording strays from the first
voicemodem demo                              the window on the loopback, talking to itself
voicemodem modes                             the voice modes, and which can be used here
voicemodem devices                           the audio devices this machine has
```

`voicemodem help` lists the options: noise, mistuning, Doppler, fading, clock
error, and where to save the audio.

## Building

```bash
cargo build --release -p app
```

The program is `target/release/voicemodem.exe`. `.cargo/config.toml` links
the C runtime in, so it runs on a machine with no developer tools. CI runs:

```bash
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
```

The release workflow builds the exe from a `v*` tag and checks that it
imports no C runtime.

## Licence

GPL-3.0-or-later. `crates/dsp` and `crates/line` come from BinModem, under the
same licence. Codec 2 is David Rowe's, in Matt Weeks's pure-Rust port
(`codec2` crate, LGPL-2.1). candle is MIT/Apache-2.0. EnCodec's weights are
not included (see above).
