First release: digital voice through a radio's voice channel.

- **Two profiles**, chosen per transmission and both listened for at once: narrow for SSB (1600 baud on 1500 Hz) and wide for FM (2400 baud on 1800 Hz).
- **BPSK, QPSK and 8PSK** at code rates 1/2, 2/3 and 3/4. Every burst opens with a DBPSK preamble that carries its own timing, carrier offset and mode.
- **Codec2** 1200 to 3200 bit/s. **EnCodec** 1.5 and 3 kbit/s modes, once Meta's weights file is downloaded (see the README).
- **Receiver**: BinModem's QAM core, trained by least squares on each preamble. It follows mistuning, satellite Doppler, fading and sound-card clock drift.
- **The window**: push to talk (button, Space or latch), constellation, spectrum and a log. Or run a loopback with no radio to hear the modes first.
- **Headless**: `voicemodem tx`, `rx`, `selftest`, `loop`, `modes` and `devices`.
