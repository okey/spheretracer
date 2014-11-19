#![macro_escape]
use std::fmt::{Show,Result,Formatter};

#[deriving(Clone)]
pub struct Vec4 {
    pub x: f64,
    pub y: f64,
    pub z: f64,
    pub w: f64
}

impl Show for Vec4 {
    fn fmt(&self, f: &mut Formatter) -> Result {
        write!(f, "(x:{}, y:{}, z:{}, w:{})", self.x, self.y, self.z, self.w)
    }
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

#[allow(dead_code)]
pub fn cross(v: &Vec4, w: &Vec4) -> Vec4 {
    let s1 = v.y * w.z - v.z * w.y;
    let s2 = v.z * w.x - v.x * w.z;
    let s3 = v.x * w.y - v.y * w.x;
    vector4!(s1 s2 s3 0.0)
}

pub fn dot(v: &Vec4, w: &Vec4) -> f64 {
  v.x * w.x + v.y * w.y + v.z * w.z
}

pub fn magnitude(v: &Vec4) -> f64 {
  dot(v, v).sqrt()
}

pub fn normalise(v: &Vec4) -> Vec4 {
  scale_by(v, 1.0 / magnitude(v))
}

pub fn scale_by(v: &Vec4, f: f64) -> Vec4 {
  Vec4 { x: v.x * f, y: v.y * f, z: v.z * f, w: v.w }
}

pub fn as_divisor(f: f64, v: &Vec4) -> Vec4 {
  Vec4 { x: f / v.x, y: f / v.y, z: f / v.z, w: v.w }
}
// TODO dot, cross, normalise, magnitude, maybe some inplace ops or macros for speed
