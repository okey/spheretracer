#![macro_escape]
use std::fmt;

/*macro_rules! clamp(
  ($a:expr) => {
    if $a < 0.0 { 0.0 }
    else if $a > 1.0 { 1.0 }
    else { $a }
  };
)*/

macro_rules! colour(
  ($r:expr $g:expr $b:expr) => { Colour { red: clamp($r),
                                          green: clamp($g),
                                          blue: clamp($b) } };
  ($a:expr) => { Colour { red: clamp($a[0]),
                          green: clamp($a[1]),
                          blue: clamp($a[2]) } };
  () => { Colour { red: 0.0, green: 0.0, blue: 0.0 } };
)

#[warn(dead_code)]
pub const BLACK: Colour = colour!();
#[warn(dead_code)]
pub const WHITE: Colour = Colour { red: 1.0, green: 1.0, blue: 1.0 };


#[deriving(Clone)]
pub struct Colour {
    pub red: f32,
    pub green: f32,
    pub blue: f32
    // no alpha for now
}

pub fn clamp(f: f32) -> f32 {
  if f < 0.0 { 0.0 }
  else if f > 1.0 { 1.0 }
  else { f }
}  

// TODO operator overloading
pub fn colour_scale(c: &Colour, s: f32) -> Colour {
  colour!(c.red*s c.green*s c.blue*s)
}

pub fn colour_multiply(c: &Colour, d: &Colour) -> Colour {
  colour!(c.red*d.red c.green*d.green c.blue*d.blue)
}

pub fn colour_add(c: &Colour, d: &Colour) -> Colour {
  colour!(c.red+d.red c.green+d.green c.blue+d.blue)
}

impl fmt::Show for Colour {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "#{:02X}{:02X}{:02X}",
               (self.red * 255.0) as u8,
               (self.green * 255.0) as u8,
               (self.blue * 255.0) as u8)
    }
}
