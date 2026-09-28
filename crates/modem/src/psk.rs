//! The three payload constellations, their Gray labels and their soft
//! demapping.
//!
//! BPSK and 8PSK have their points at `2 pi k / M` from +1. QPSK has its
//! points on the diagonals, at 45, 135, 225 and 315 degrees: the way the
//! textbooks, DVB-S and V.34's inner four draw it, with the axes as its
//! decision lines.
//!
//! The pilots go out on the constellation's first point and its opposite
//! ([`Modulation::pilot`]): +-1, or for QPSK +-(1 + j)/sqrt 2. They have to
//! be points of the constellation the receiver is slicing against, since it
//! slices them with the data. A pilot off the constellation would read to it
//! as an error the size of the distance to the nearest point, sixteen symbols
//! in a row, every slot, and BinModem's core would call that a lost signal.
//! The preamble is +-1 whatever follows it; the receiver trains on it as
//! known symbols, and slices nothing until it is over.

use std::f64::consts::{FRAC_PI_4, TAU};

use dsp::Complex;
use dsp::qam::{Constellation, Slicer};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Modulation {
    Bpsk,
    Qpsk,
    Psk8,
}

impl Modulation {
    pub const ALL: [Modulation; 3] = [Modulation::Bpsk, Modulation::Qpsk, Modulation::Psk8];

    pub fn bits(self) -> usize {
        match self {
            Modulation::Bpsk => 1,
            Modulation::Qpsk => 2,
            Modulation::Psk8 => 3,
        }
    }

    pub fn points(self) -> usize {
        1 << self.bits()
    }

    pub fn code(self) -> u8 {
        match self {
            Modulation::Bpsk => 0,
            Modulation::Qpsk => 1,
            Modulation::Psk8 => 2,
        }
    }

    pub fn from_code(code: u8) -> Option<Self> {
        match code {
            0 => Some(Modulation::Bpsk),
            1 => Some(Modulation::Qpsk),
            2 => Some(Modulation::Psk8),
            _ => None,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Modulation::Bpsk => "BPSK",
            Modulation::Qpsk => "QPSK",
            Modulation::Psk8 => "8PSK",
        }
    }

    pub fn parse(text: &str) -> Option<Self> {
        match text.trim().to_ascii_lowercase().as_str() {
            "bpsk" | "2" => Some(Modulation::Bpsk),
            "qpsk" | "4" => Some(Modulation::Qpsk),
            "8psk" | "psk8" | "8" => Some(Modulation::Psk8),
            _ => None,
        }
    }

    /// The angle of point nought: 45 degrees for QPSK, on the diagonal, and
    /// nought, on +1, for the others.
    pub fn offset(self) -> f64 {
        match self {
            Modulation::Qpsk => FRAC_PI_4,
            Modulation::Bpsk | Modulation::Psk8 => 0.0,
        }
    }

    /// Point `k`, at angle `offset + 2 pi k / M`.
    pub fn point(self, k: usize) -> Complex {
        Complex::from_polar(1.0, self.offset() + TAU * k as f64 / self.points() as f64)
    }

    /// The point a pilot chip of +1 goes out as. A chip of -1 goes out as its
    /// opposite, which is a point too, `M` being even.
    pub fn pilot(self) -> Complex {
        self.point(0)
    }

    /// The points in angle order.
    pub fn constellation(self) -> Vec<Complex> {
        (0..self.points()).map(|k| self.point(k)).collect()
    }

    /// A slicer for BinModem's core, points in angle order.
    pub fn slicer(self) -> Slicer {
        Slicer::table(Constellation::new(self.constellation()))
    }

    /// The point carrying `label`, `bits()` bits read most significant first.
    ///
    /// Gray: the points either side of any point differ from it in one bit,
    /// so the commonest error -- to a neighbour -- costs one bit.
    pub fn map(self, label: usize) -> Complex {
        self.point(ungray(label))
    }

    /// The point index a received value lies nearest to.
    pub fn nearest(self, z: Complex) -> usize {
        let m = self.points() as f64;
        (((z.arg() - self.offset()) / TAU * m).round().rem_euclid(m)) as usize % self.points()
    }

    /// Max-log soft values for each of a symbol's bits, most significant
    /// first, pushed onto `out`: `(d1 - d0) / noise`, where `d0` and `d1` are
    /// the squared distances to the nearest point whose label has that bit
    /// nought and one.
    pub fn demap(self, z: Complex, noise: f64, out: &mut Vec<f64>) {
        let bits = self.bits();
        let mut d = [f64::INFINITY; 8];
        for (k, slot) in d.iter_mut().enumerate().take(self.points()) {
            *slot = (z - self.point(k)).norm_sqr();
        }
        let scale = 1.0 / noise.max(1e-6);
        for bit in (0..bits).rev() {
            let (mut d0, mut d1) = (f64::INFINITY, f64::INFINITY);
            for (k, &dk) in d.iter().enumerate().take(self.points()) {
                if (gray(k) >> bit) & 1 == 0 {
                    d0 = d0.min(dk);
                } else {
                    d1 = d1.min(dk);
                }
            }
            out.push((d1 - d0) * scale);
        }
    }
}

fn gray(k: usize) -> usize {
    k ^ (k >> 1)
}

fn ungray(mut g: usize) -> usize {
    let mut k = 0;
    while g != 0 {
        k ^= g;
        g >>= 1;
    }
    k
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn neighbours_differ_by_one_bit() {
        for m in Modulation::ALL {
            let n = m.points();
            for k in 0..n {
                let (a, b) = (gray(k), gray((k + 1) % n));
                assert_eq!((a ^ b).count_ones(), 1, "{} points {k} and {}", m.label(), (k + 1) % n);
                assert_eq!(ungray(gray(k)), k);
            }
        }
    }

    #[test]
    fn every_constellation_holds_its_pilots() {
        for m in Modulation::ALL {
            let points = m.constellation();
            for target in [m.pilot(), -m.pilot()] {
                assert!(points.iter().any(|p| (*p - target).abs() < 1e-12), "{} lacks {target:?}", m.label());
            }
        }
    }

    #[test]
    fn qpsk_is_on_the_diagonals_and_the_others_on_plus_one() {
        let h = std::f64::consts::FRAC_1_SQRT_2;
        for (k, (re, im)) in [(h, h), (-h, h), (-h, -h), (h, -h)].into_iter().enumerate() {
            assert!((Modulation::Qpsk.point(k) - Complex::new(re, im)).abs() < 1e-12, "QPSK point {k}");
        }
        assert!((Modulation::Bpsk.pilot() - Complex::ONE).abs() < 1e-12);
        assert!((Modulation::Psk8.pilot() - Complex::ONE).abs() < 1e-12);
    }

    #[test]
    fn demapping_a_clean_point_gives_its_label() {
        for m in Modulation::ALL {
            for label in 0..m.points() {
                let mut soft = Vec::new();
                m.demap(m.map(label), 0.1, &mut soft);
                let decided = soft.iter().fold(0usize, |acc, &l| (acc << 1) | usize::from(l < 0.0));
                assert_eq!(decided, label, "{}", m.label());
                assert_eq!(m.nearest(m.map(label)), ungray(label));
            }
        }
    }
}
