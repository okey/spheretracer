#![macro_use]
use std::fmt;
use std::ops::{Add,Mul,Index};


// Colour module - RGB [0.0,1.0]

// Types
// Each channel is intended to be in [0.0,1.0]. Call colour_clamp to enforce this.
// 32-bit floats are used for direct compatibility with GLfloat
#[derive(PartialEq,Copy,Clone,Debug)]
pub struct Colour {
    pub red: f32,
    pub green: f32,
    pub blue: f32
    // no alpha for now
}

// Macros
macro_rules! colour(
  ($r:expr, $g:expr, $b:expr) => { Colour { red:   $r,
                                          green: $g,
                                          blue:  $b } };
  ($a:expr) => { Colour { red:  $a[0],
                          green:$a[1],
                          blue: $a[2] } };
  () => { Colour { red: 0.0, green: 0.0, blue: 0.0 } };
);

// Constants
pub const BLACK: Colour = colour!();
pub const WHITE: Colour = colour!(1.0, 1.0, 1.0);
pub const CHANNELS: usize = 3;


// Public functions
pub fn from_slice(v: &[f32]) -> Colour { // what about using a macro for consistency?
  assert!(v.len() == 3);
  let c = colour!(v);
  c.clamp()
}


// Custom traits and their implementations
trait Apply {
  fn apply<F>(&self, f: F) -> Self
    where F : Fn(f32) -> f32;
}

trait Combine<T> {
  fn combine<F>(&self, b: &Self, f: F) -> Self
    where F : Fn(T, T) -> T;
}

pub trait Clamp {
  fn clamp(&self) -> Self;
}

impl Apply for Colour {
  fn apply<F>(&self, f: F) -> Colour
    where F : Fn(f32) -> f32 {
    colour!(f(self.red), f(self.green), f(self.blue))
  }
}

impl Combine<f32> for Colour {
  fn combine<F>(&self, b: &Colour, f: F) -> Colour
    where F : Fn(f32, f32) -> f32{
    colour!(f(self.red, b.red), f(self.green, b.green), f(self.blue, b.blue))
  }
}

impl Clamp for Colour {
  fn clamp(&self) -> Colour {
    self.apply(|c| if c > 1.0 { 1.0 } else if c < 0.0 { 0.0} else { c })
  }
}

// Standard trait implementations
impl Mul<f32> for Colour {
  type Output = Colour;

  fn mul(self, rhs: f32) -> Colour {
    self.apply(|c| c * rhs)
  }
}

impl Mul for Colour {
  type Output = Colour;

  fn mul(self, rhs: Colour) -> Colour {
    self.combine(&rhs, |a, b| a * b)
  }
}

impl Add for Colour {
  type Output = Colour;

  fn add(self, rhs: Colour) -> Colour {
    self.combine(&rhs, |a, b| a + b)
  }
}

impl Index<usize> for Colour {
  type Output = f32;

  fn index<'a>(&'a self, _idx: usize) -> &'a f32 {
    assert!(_idx < CHANNELS);
    match _idx {
      0 => &self.red, 1 => &self.green, 2 => &self.blue,
      _ => panic!("Index {} out of bounds", _idx)
    }
  }
}

impl fmt::Display for Colour {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "#{:02X}{:02X}{:02X}",
               (self.red * 255.0) as u8,
               (self.green * 255.0) as u8,
               (self.blue * 255.0) as u8)
    }
}

// Unit tests
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
    let red = colour!(1.0, 0.0, 0.0);
    let blue = colour!(0.0, 0.0, 1.0);
    let purple = colour!(1.0, 0.0, 1.0);

    assert!(red + blue == purple);
  }

  #[test]
  fn colour_mul() {
    let grey = colour!(0.5, 0.5, 0.5);

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
