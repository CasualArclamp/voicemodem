//! What the window remembers from one run to the next, in
//! `%APPDATA%\voicemodem\settings.txt` as `key = value` lines.

use std::path::PathBuf;

#[derive(Debug, Clone, PartialEq)]
pub struct Settings {
    pub radio_in: String,
    pub radio_out: String,
    pub mic: String,
    pub speaker: String,
    pub loopback: bool,
    pub loopback_snr: f64,
    pub mode: String,
    pub text: String,
    pub level: f64,
    pub full_duplex: bool,
    pub latch: bool,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            radio_in: String::new(),
            radio_out: String::new(),
            mic: String::new(),
            speaker: String::new(),
            // With nothing chosen yet, the loopback is the one thing that
            // works on any machine.
            loopback: true,
            loopback_snr: 12.0,
            mode: "narrow-robust".into(),
            text: String::new(),
            level: -12.0,
            full_duplex: false,
            latch: false,
        }
    }
}

fn path() -> Option<PathBuf> {
    let base = std::env::var_os("APPDATA").or_else(|| std::env::var_os("HOME"))?;
    Some(PathBuf::from(base).join("voicemodem").join("settings.txt"))
}

impl Settings {
    pub fn load() -> Self {
        let mut s = Self::default();
        let Some(text) = path().and_then(|p| std::fs::read_to_string(p).ok()) else { return s };
        for line in text.lines() {
            let Some((key, value)) = line.split_once('=') else { continue };
            let value = value.trim().to_string();
            let flag = value == "true";
            let number = value.parse::<f64>().ok();
            match key.trim() {
                "radio_in" => s.radio_in = value,
                "radio_out" => s.radio_out = value,
                "mic" => s.mic = value,
                "speaker" => s.speaker = value,
                "loopback" => s.loopback = flag,
                "loopback_snr" => s.loopback_snr = number.unwrap_or(s.loopback_snr),
                "mode" => s.mode = value,
                "text" => s.text = value,
                "level" => s.level = number.unwrap_or(s.level),
                "full_duplex" => s.full_duplex = flag,
                "latch" => s.latch = flag,
                _ => {}
            }
        }
        s
    }

    pub fn save(&self) {
        let Some(path) = path() else { return };
        if let Some(dir) = path.parent() {
            let _ = std::fs::create_dir_all(dir);
        }
        let text = format!(
            "radio_in = {}\nradio_out = {}\nmic = {}\nspeaker = {}\nloopback = {}\nloopback_snr = {}\nmode = {}\ntext = {}\nlevel = {}\nfull_duplex = {}\nlatch = {}\n",
            self.radio_in,
            self.radio_out,
            self.mic,
            self.speaker,
            self.loopback,
            self.loopback_snr,
            self.mode,
            self.text,
            self.level,
            self.full_duplex,
            self.latch
        );
        let _ = std::fs::write(path, text);
    }
}
