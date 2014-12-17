#![macro_escape]
use std::fmt::{Show,Result,Formatter};
use std::num::Float;
use mat4;

// This module is for a three dimensional vector implementation that includes a homogeneous
// coordinate for use with transforms.

// Vec3 could be an array type... but then I can't implement traits without either containing it
// or using newtype. The former is ugly and the latter is tedious because
// Consider rewrite when newtype stuff gets added or if Rust adds properties
#[deriving(PartialEq,Copy)]
pub struct Vec3 {
  pub x: f64,
  pub y: f64,
  pub z: f64,
  pub w: f64
}

// Construction macros
macro_rules! vector_h(
  ($x:expr $y:expr $z:expr $w:expr) => { Vec3 { x: $x, y: $y, z: $z, w: $w } };
  ($a:expr $w:expr) => { Vec3 { x: $a[0], y: $a[1], z: $a[2], w: $w } };
  ($a:expr) => { Vec3 { x: $a[0], y: $a[1], z: $a[2], z: $a[3] } };
)

macro_rules! vector(
  ($x:expr $y:expr $z:expr) => { vector_h!($x $y $z 0f64) };
  ($a:expr) => { vector_h!($a 0f64) };
  () => { vector_h!(0f64 0f64 0f64 0f64) };
)

macro_rules! point(
  ($x:expr $y:expr $z:expr) => { vector_h!($x $y $z 1f64) };
  ($a:expr) => { vector_h!($a 1f64) };
  () => { vector_h!(0f64 0f64 0f64 1f64) };
)

// Public functions
pub fn parametric_position(u: &Vec3, v: &Vec3, t: f64) -> Vec3{
  *u + (*v * t)
}

// Standard trait implementations
impl Show for Vec3 {
  fn fmt(&self, f: &mut Formatter) -> Result {
    write!(f, "(x:{}, y:{}, z:{}, w:{})", self.x, self.y, self.z, self.w)
  }
}

impl Index<uint, f64> for Vec3 {
  fn index<'a>(&'a self, _idx: &uint) -> &'a f64 {
    match *_idx {
      0 => &self.x, 1 => &self.y, 2 => &self.z, 3 => &self.w,
      _ => panic!("Index {} out of bounds", _idx)
    }
  }
}

impl Add<Vec3, Vec3> for Vec3 {
  fn add(self, rhs: Vec3) -> Vec3 {
    Vec3 { x: self.x + rhs.x,
           y: self.y + rhs.y,
           z: self.z + rhs.z,
           w: if self.w + rhs.w == 1.0 { 1.0 } else { 0.0 }
    }
    // vector + vector = vector
    // vector + point  = point
    // point  + point  = vector
  }
}

impl Sub<Vec3, Vec3> for Vec3 {
  fn sub(self, rhs: Vec3) -> Vec3 {
    Vec3 { x: self.x - rhs.x,
           y: self.y - rhs.y,
           z: self.z - rhs.z,
           w: if self.w + rhs.w == 1.0 { 1.0 } else { 0.0 }
    }
  }
}

impl Mul<f64, Vec3> for Vec3 {
  fn mul(self, rhs: f64) -> Vec3 {
    let f = rhs;
    Vec3 { x: self.x * f, y: self.y * f, z: self.z * f, w: self.w }
  }
}

// Custom traits
pub trait Apply {
  fn apply(&self, f: |f64| -> f64) -> Self;
}

// vec / f64 is semantically different, and f64 / vec doesn't work because the compiler
// expects f64 / f64
pub trait AsDivisorOf {
  fn as_divisor_of(&self, f: f64) -> Self;
}

pub trait DotProduct {
  fn dot(&self, rhs: &Self) -> f64;
}

pub trait CrossProduct {
  fn cross(&self, rhs: &Self) -> Self;
}

pub trait Magnitude {
  fn magnitude(&self) -> f64;
}

pub trait Normalise {
  fn normalise(&self) -> Self;
}

pub trait Transform {
  fn transform(&self, t: &mat4::Matrix) -> Self;
}

pub trait AsVector {
  fn as_vector(&self) -> Self;
}

pub trait AsPoint {
  fn as_point(&self) -> Self;
}

// Custom trait implementations
impl Apply for Vec3 {
  fn apply(&self, f: |f64| -> f64) -> Vec3 {
    Vec3 { x: f(self.x), y: f(self.y), z: f(self.z), w: self.w }
  }
}

impl AsDivisorOf for Vec3 {
  fn as_divisor_of(&self, f: f64) -> Vec3 {
    self.apply(|c| f / c)
  }
}

impl DotProduct for Vec3 {
  fn dot(&self, rhs: &Vec3) -> f64 {
    self.x * rhs.x + self.y * rhs.y + self.z * rhs.z
  }
}

impl CrossProduct for Vec3 {
  fn cross(&self, rhs: &Vec3) -> Vec3 {
    Vec3 {
      x: self.y * rhs.z - self.z * rhs.y,
      y: self.z * rhs.x - self.x * rhs.z,
      z: self.x * rhs.y - self.y * rhs.x,
      w: 0.0
    }
  }
}

impl Magnitude for Vec3 {
  fn magnitude(&self) -> f64 {
    self.dot(self).sqrt()
  }
}

impl Normalise for Vec3 {
  fn normalise(&self) -> Vec3 {
    self.mul(1.0 / self.magnitude())
  }
}

impl Transform for Vec3 {
  fn transform(&self, t: &mat4::Matrix) -> Vec3 {
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

impl AsVector for Vec3 {
  fn as_vector(&self) -> Vec3 {
    return Vec3 { x: self.x, y: self.y, z: self.z, w: 0.0 }
  }
}

impl AsPoint for Vec3 {
  fn as_point(&self) -> Vec3 {
    return Vec3 { x: self.x, y: self.y, z: self.z, w: 1.0 }
  }
}

// Unit tests
#[cfg(test)]
mod test {
  use super::*;
  use std::f64::consts;
  use mat4;
  
  #[test]
  fn vec3_equality() {
    let a  = vector!(1.0 2.0 3.0);
    let a2 = vector!(1.0 2.0 3.0);
    let a3 = point!(1.0 2.0 3.0);
    let b  = vector!(4.0 5.0 6.0);
    
    assert!(a == a2);
    assert!(a != b);
    assert!(a != a3);
  }

  #[test]
  fn vec3_simple_operators() {
    let a  = vector!(1.0 2.0 3.0);
    let b  = vector!(1.0 2.0 3.0);
    let c  = vector!(10.0 10.0 10.0); 
    let d  = vector!(1.0 1.0 1.0);
    
    assert!(a[0] == 1.0 && a[1] == 2.0 && a[2] == 3.0);
    assert!(a + b == a * 2.0);
    assert!(a + b - a == b);
    assert!(c.as_divisor_of(10.0) == d);
  }

  #[test]
  fn vec3_simple_algebra() {
    let a  = vector!(3.0 4.0 0.0);
    let b  = vector!(1.0 0.0 0.0);
    let c  = vector!(0.0 1.0 0.0);
    let d  = vector!(0.0 0.0 1.0);
    
    assert!(a.magnitude() == 5.0);
    assert!(a.normalise().magnitude() == 1.0);
    assert!(a.dot(&a) == a.magnitude() * a.magnitude());
    assert!(b.cross(&c) == d);
    assert!(c.cross(&d) == b);
    assert!(d.cross(&b) == c);
  }

  #[test]
  fn vec3_homogeneous() {
    let a = vector_h!(1.0 2.0 3.0 4.0);
    let b = vector!(4.0 5.0 6.0);
    let c = a.as_point();
    
    assert!(a.as_vector().w == 0.0);
    assert!(c.w == 1.0);
    assert!((c - b).w == 1.0);
    assert!((c + b).w == 1.0);
    assert!((b - b).w == 0.0);
    assert!((b + b).w == 0.0);
    assert!((c - c).w == 0.0);
    assert!((c + c).w == 0.0);
  }

  #[test]
  fn vec3_transform() {
    let a = point!(1.0 2.0 3.0);
    let b = vector!(1.0 2.0 3.0);

    let m = mat4::new_scale(&vector!(3.0 3.0 3.0));
    let n = mat4::new_translation(&vector!(3.0 3.0 3.0));
    let o = mat4::new_rotation(&vector!(1.0 0.0 0.0), consts::PI / 2.0);

    let c = a.transform(&o).transform(&n).transform(&m);
    let d = b.transform(&o).transform(&m);

    let m9 = -9.0;
    assert!(c == point!(12.0 0.0 15.0));
    assert!(d == vector!(3.0 m9 6.0));
  }
}

