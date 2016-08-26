#![macro_use]
use std::fmt;

use colour;
use colour::Colour;
use mat4;
use vec3::Vec3;

// Types
#[derive(Debug)]
pub struct Light {
    pub position: Vec3,
    pub colour: Colour
}

#[derive(Copy,Clone,Debug)]
pub struct Material {
    pub diffuse: Colour,
    pub mirror: Colour,
    pub phong: Colour,
    pub phong_n: u8
}

#[derive(Debug)]
pub struct Sphere {
    pub position: Vec3,
    pub inner: Material,
    pub outer: Material,
    pub radius: f64,
    pub transform: mat4::Matrix,
    pub inverse_t: mat4::Matrix
}

#[derive(Debug)]
pub struct Scene {
    pub image_size: (u16, u16),
    pub ambient: Colour,
    pub background: Colour,
    pub lights: Vec<Light>,
    pub spheres: Vec<Sphere>
}

// Macrossc
macro_rules! material(
    () => { Material { diffuse: colour!(1.0, 1.0, 1.0),
                       mirror: colour!(),
                       phong: colour!(1.0, 1.0, 1.0), phong_n: 10 } };
);

macro_rules! sphere(
    () => {
    Sphere { position: point!(), inner: material!(), outer: material!(), radius: 0f64,
             transform: mat4::identity(), inverse_t: mat4::identity() }
    };
);

// Public functions
pub fn new_material(d: &[f32], m: &[f32], p: &[f32], n: u8) -> Material {
    Material { diffuse: colour::from_slice(d),
               mirror: colour::from_slice(m),
               phong: colour::from_slice(p),
               phong_n: n,
    }
}

// Standard trait implementations
impl fmt::Display for Light {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "\nposition {}, colour {}", self.position, self.colour)
    }
}

impl fmt::Display for Material {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{} {} {} {}", self.diffuse, self.mirror, self.phong, self.phong_n)
    }
}

impl fmt::Display for Sphere {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "\n{} {} {} {}", self.position, self.inner, self.outer, self.radius)
    }
}

impl fmt::Display for Scene {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "\nimage_size {:?}\nambient {}\nbackground {}\nlights {} {:?}\nspheres {} {:?}",
               self.image_size, self.ambient, self.background,
               self.lights.len(), self.lights, self.spheres.len(), self.spheres)
    }
}
