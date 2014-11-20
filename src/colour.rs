#![macro_escape]
use std::fmt;

pub const BLACK: Colour = Colour { red: 0, green: 0, blue: 0 };
pub const WHITE: Colour = Colour { red: 255, green: 255, blue: 255 };


#[deriving(Clone)]
pub struct Colour {
    pub red: u8,
    pub green: u8,
    pub blue: u8,
    // no alpha for now
}

impl fmt::Show for Colour {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "#{:X}{:X}{:X}", self.red, self.green, self.blue)
    }
}

macro_rules! colour(
    ($r:expr $g:expr $b:expr) => { Colour { red: $r, green: $g, blue: $b } };
    ($a:expr) => { Colour { red: $a[0], green: $a[1], blue: $a[2] } };
    () => { Colour { red: 0, green: 0, blue: 0 } };
)
