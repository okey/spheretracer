#![macro_escape]
use std::fmt;

use colour::Colour;
use mat4;
use vec4::Vec4;

pub struct Light {
    pub position: Vec4,
    pub colour: Colour
}

pub struct Material {
    pub diffuse: Colour,
    pub mirror: Colour,
    pub phong: Colour,
    pub phong_n: u8
}

pub struct Sphere {
    pub position: Vec4,
    pub inner: Material,
    pub outer: Material,
    pub radius: f64,
    pub transform: mat4::Matrix,
    pub inverse_t: mat4::Matrix
}

pub struct Scene {
    pub image_size: (u16, u16),
    pub ambient: Colour,
    pub background: Colour,
    pub lights: Vec<Light>,
    pub spheres: Vec<Sphere>
}



impl fmt::Show for Light {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "\nposition {}, colour {}", self.position, self.colour)
    }
}

impl fmt::Show for Material {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{} {} {} {}", self.diffuse, self.mirror, self.phong, self.phong_n)
    }
}

impl fmt::Show for Sphere {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "\n{} {} {} {}", self.position, self.inner, self.outer, self.radius)
    }
}

impl fmt::Show for Scene {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "\nimage_size {}\nambient {}\nbackground {}\nlights {} {}\nspheres {} {}",
               self.image_size, self.ambient, self.background, self.lights.len(), self.lights, self.spheres.len(), self.spheres)
    }
}

macro_rules! material(
    () => { Material { diffuse: colour!(), mirror: colour!(), phong: colour!(), phong_n: 1 } };
)

macro_rules! sphere(
    () => {
    Sphere { position: point!(), inner: material!(), outer: material!(), radius: 0f64,
             transform: mat4::identity(), inverse_t: mat4::identity() }
    };
)

pub fn make_colour(v: &[f32]) -> Colour { // what about using a macro for consistency?
  assert!(v.len() == 3);
  colour!(v)
}

pub fn make_material(d: &[f32], m: &[f32], p: &[f32], n: u8) -> Material {
    Material { diffuse: make_colour(d), mirror: make_colour(m), phong: make_colour(p), phong_n: n }
}
