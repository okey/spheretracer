#![macro_escape]
use std::fmt;

macro_rules! colour(
  ($r:expr $g:expr $b:expr) => { Colour { red:   $r,
                                          green: $g,
                                          blue:  $b } };
  ($a:expr) => { Colour { red:  $a[0],
                          green:$a[1],
                          blue: $a[2] } };
  () => { Colour { red: 0.0, green: 0.0, blue: 0.0 } };
)

pub const BLACK: Colour = colour!();
pub const WHITE: Colour = colour!(1.0 1.0 1.0);

// Each channel is intended to be in [0.0,1.0]. Call colour_clamp to enforce this
// 32-bit floats are used for direct compatibility with GLfloat
#[deriving(PartialEq)]
pub struct Colour {
    pub red: f32,
    pub green: f32,
    pub blue: f32
    // no alpha for now
}

/* Custom traits and their implementations */
trait Apply {
  fn apply(&self, f: |f32| -> f32) -> Self;
}

trait Combine<T> {
  fn combine(&self, b: &Self, f: |T, T| -> T) -> Self;
}

pub trait Clamp {
  fn clamp(&self) -> Self;
}

impl Apply for Colour {
  fn apply(&self, f: |f32| -> f32) -> Colour {
    colour!(f(self.red) f(self.green) f(self.blue))
  }
}

impl Combine<f32> for Colour {
  fn combine(&self, b: &Colour, f: |f32, f32| -> f32) -> Colour {
    colour!(f(self.red, b.red) f(self.green, b.green) f(self.blue, b.blue))
  }
}

impl Clamp for Colour {
  fn clamp(&self) -> Colour {
    self.apply(|c| if c > 1.0 { 1.0 } else if c < 0.0 { 0.0} else { c })
  }
}

/* Standard trait implementations */
impl Mul<f32, Colour> for Colour {
  fn mul(&self, rhs: &f32) -> Colour {
    self.apply(|c| c * *rhs)
  }
}

impl Mul<Colour, Colour> for Colour {
  fn mul(&self, rhs: &Colour) -> Colour {
    self.combine(rhs, |a, b| a * b)
  }
}

impl Add<Colour, Colour> for Colour {
  fn add(&self, rhs: &Colour) -> Colour {
    self.combine(rhs, |a, b| a + b)
  }
}

impl fmt::Show for Colour {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "#{:02X}{:02X}{:02X}",
               (self.red * 255.0) as u8,
               (self.green * 255.0) as u8,
               (self.blue * 255.0) as u8)
    }
}

/* Unit tests */
#[cfg(test)]
mod test {
  
  use super::*;
  use super::{Apply};

  #[test]
  fn colour_eq() {
    assert!(BLACK != WHITE);
  }

  #[test]
  fn colour_add() {
    let red = colour!(1.0 0.0 0.0);
    let blue = colour!(0.0 0.0 1.0);
    let purple = colour!(1.0 0.0 1.0);

    assert!(red + blue == purple);
  }

  #[test]
  fn colour_mul() {
    let grey = colour!(0.5 0.5 0.5);

    assert!(grey == WHITE * 0.5);
  }

  #[test]
  fn colour_clamp() {
    let overload = WHITE + WHITE;
    let underload = BLACK.apply(|c| c - 1.0);

    assert!(overload.clamp() == WHITE);
    assert!(underload.clamp() == BLACK);
  }
}
