#![macro_escape]
use std::fmt::{Show,Result,Formatter};
use mat4;

// TODO rename Vec3 as it is really vec3 + homogenous coord
// TODO fn that takes a lambda to apply to each elem of vec

#[deriving(Clone)]
pub struct Vec4 {
  pub x: f64,
  pub y: f64,
  pub z: f64,
  pub w: f64
}

macro_rules! vector4(
  ($x:expr $y:expr $z:expr $w:expr) => { Vec4 { x: $x, y: $y, z: $z, w: $w } };
  ($a:expr $w:expr) => { Vec4 { x: $a[0], y: $a[1], z: $a[2], w: $w } };
  ($a:expr) => { Vec4 { x: $a[0], y: $a[1], z: $a[2], z: $a[3] } };
)

macro_rules! vector(
  ($x:expr $y:expr $z:expr) => { vector4!($x $y $z 0f64) };
  ($a:expr) => { vector4!($a 0f64) };
  () => { vector4!(0f64 0f64 0f64 0f64) };
)

macro_rules! point(
  ($x:expr $y:expr $z:expr) => { vector4!($x $y $z 1f64) };
  ($a:expr) => { vector4!($a 1f64) };
  () => { vector4!(0f64 0f64 0f64 1f64) };
)

impl Show for Vec4 {
  fn fmt(&self, f: &mut Formatter) -> Result {
    write!(f, "(x:{}, y:{}, z:{}, w:{})", self.x, self.y, self.z, self.w)
  }
}

// This is probably slower than a copy to array... could just make Vec4 an array type
// But then we can't implement traits for it (yet) so I'd have to /contain/ it... which sucks too
// since Rust doesn't have properties (yet?)
impl Index<uint, f64> for Vec4 {
  fn index<'a>(&'a self, _idx: &uint) -> &'a f64 {
    match *_idx {
      0 => &self.x, 1 => &self.y, 2 => &self.z, 3 => &self.w,
      _ => panic!("Index {} out of bounds", _idx)
    }
  }
}

impl Add<Vec4, Vec4> for Vec4 {
  fn add(&self, rhs: &Vec4) -> Vec4 {
    Vec4 { x: self.x + rhs.x,
           y: self.y + rhs.y,
           z: self.z + rhs.z,
           w: if self.w + rhs.w == 1.0 { 1.0 } else { 0.0 }
    }
    // vector + vector = vector
    // vector + point  = point
    // point  + point  = ??? but let's assume a vector for now
  }
}

impl Sub<Vec4, Vec4> for Vec4 {
  fn sub(&self, rhs: &Vec4) -> Vec4 {
    Vec4 { x: self.x - rhs.x,
           y: self.y - rhs.y,
           z: self.z - rhs.z,
           w: if self.w + rhs.w == 1.0 { 1.0 } else { 0.0 }
    }
  }
}

impl Mul<f64, Vec4> for Vec4 {
  fn mul(&self, rhs: &f64) -> Vec4 {
    let f = *rhs;
    Vec4 { x: self.x * f, y: self.y * f, z: self.z * f, w: self.w }
  }
}

//pub mod vtraits {
//  use super::Vec4;
pub trait Apply {
  fn apply(&self, f: |f64| -> f64) -> Self;
}

impl Apply for Vec4 {
  fn apply(&self, f: |f64| -> f64) -> Vec4 {
    Vec4 { x: f(self.x), y: f(self.y), z: f(self.z), w: self.w }
  }
}

// vec / f64 is semantically different, and f64 / vec doesn't work because the compiler
// expects f64 / f64
pub trait AsDivisorOf {
  fn as_divisor_of(&self, f: f64) -> Self;
}

impl AsDivisorOf for Vec4 {
  fn as_divisor_of(&self, f: f64) -> Vec4 {
    self.apply(|c| f / c)
  }
}
//}

pub trait DotProduct {
  fn dot(&self, rhs: &Self) -> f64;
}

impl DotProduct for Vec4 {
  fn dot(&self, rhs: &Vec4) -> f64 {
    self.x * rhs.x + self.y * rhs.y + self.z * rhs.z
  }
}

pub trait CrossProduct {
  fn cross(&self, rhs: &Self) -> Self;
}

impl CrossProduct for Vec4 {
  fn cross(&self, rhs: &Vec4) -> Vec4 {
    Vec4 {
      x: self.y * rhs.z - self.z * rhs.y,
      y: self.z * rhs.x - self.x * rhs.z,
      z: self.x * rhs.y - self.y * rhs.x,
      w: 0.0
    }
  }
}

pub trait Magnitude {
  fn magnitude(&self) -> f64;
}

impl Magnitude for Vec4 {
  fn magnitude(&self) -> f64 {
    self.dot(self).sqrt()
  }
}

pub trait Normalise {
  fn normalise(&self) -> Self;
}

impl Normalise for Vec4 {
  fn normalise(&self) -> Vec4 {
    self.mul(&(1.0 / self.magnitude()))
  }
}

pub trait Transform {
  fn transform(&self, t: &mat4::Matrix) -> Self;
}

impl Transform for Vec4 {
  fn transform(&self, t: &mat4::Matrix) -> Vec4 {
    let mut w = vector!();

    for c in range(0, mat4::DIM) {
      w.x += t[0][c] * self[c];
      w.y += t[1][c] * self[c];
      w.z += t[2][c] * self[c];
    }
    w.w = self.w;

    w
  }
}

pub trait AsVector {
  fn as_vector(&self) -> Self;
}

impl AsVector for Vec4 {
  fn as_vector(&self) -> Vec4 {
    return Vec4 { x: self.x, y: self.y, z: self.z, w: 0.0 }
  }
}

pub trait AsPoint {
  fn as_point(&self) -> Self;
}

impl AsPoint for Vec4 {
  fn as_point(&self) -> Vec4 {
    return Vec4 { x: self.x, y: self.y, z: self.z, w: 1.0 }
  }
}

pub fn parametric_position(u: &Vec4, v: &Vec4, t: f64) -> Vec4{
  *u + (*v * t)
}

// TODO perf testing? macros? mutability?
