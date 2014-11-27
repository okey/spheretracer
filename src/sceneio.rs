use std::f64::consts as f64consts;
use std::error::FromError;
use std::io::{File,BufferedReader,IoError};
use std::path::Path;
use std::{fmt,cmp};
use core::str::FromStr;

use mat4;
use vec3::{AsDivisorOf,Vec3}; // TODO Clean up trait usage when UFCS is implemented in Rust
use colour;
use colour::Colour;
use scene::{Scene,Light,Material,Sphere,new_material};

// Read a custom scene definition format and produce a scene structure for rendering
// The format is a bit of a kludge and this was implemented for the purpose of learning Rust.

// TODO rewrite or replace with TOML

// Types
#[deriving(Show)]
enum ErrorKind {
  FileOperationFailed,
  UnexpectedLine,
  InvalidLine,
}

#[deriving(Show)]
enum LineNumber<T> {
  Line(T),
  NoLine,
}

type ULineNumber = LineNumber<uint>;

pub struct Error {
  pub kind: ErrorKind,
  pub desc: String, // can't use &'static str if I want to pass in the contents of the failed line
  pub line: ULineNumber,
}

type ParseResult<T> = Result<T, Error>;

// Standard trait implemenations
impl FromError<IoError> for Error {
  fn from_error(e: IoError) -> Error {
    match e {
      _ => Error { kind: ErrorKind::FileOperationFailed,
                        desc: String::from_str(e.desc),
                        line: LineNumber::NoLine },
    }
  }
}

impl fmt::Show for Error {
  fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
    match self.kind {
      ErrorKind::FileOperationFailed => write!(f, "{} : {}", self.kind, self.desc),
      _ => write!(f, "{} {}: {}", self.kind, self.line, self.desc)
    }
  }
}

// Macros
macro_rules! radians(
  ($degrees:expr) => {$degrees as f64 * f64consts::PI / 180.0};)


// Public functions

// Read a scene file or return a Error
// Warning: monolithic
pub fn read_scene(filename: &Path) -> ParseResult<Scene> {
  // constrain image_size to +ve, proper window setup will take care of the rest
  const MIN_IMG_DIM: u16 = 1;

  let mut file = BufferedReader::new(File::open(filename));
  let mut line_num = 0u;
  let space_tab_re = regex!(r"[ \t]+");

  // Default scene
  let mut scene = Scene {
    image_size: (800, 600),
    ambient: colour::WHITE,
    background: colour::BLACK,
    lights: Vec::new(),
    spheres: Vec::new()
  };

  // Used to reject attempts to set object properties from outside an object
  let mut in_object = false;

  for line in file.lines() {
    let line_str = try!(line);

    // Collect non-empty tokens separated by arbitrary \t or spaces
    // but ignore the rest of the line if we encounter a #
    let tokens = space_tab_re.split(line_str.trim())
      .take_while(|&s| !s.starts_with("#"))
      .filter(|&s| !s.is_empty())
      .collect::<Vec<&str>>();

    // Skip blank lines and comments
    let n_toks = tokens.len(); // Used later for line validity checks
    if n_toks == 0 {
      line_num += 1;
      continue;
    }


    let s = tokens[0];
    match s {
      // Parse scene properties
      
      // Not using guards since we check the result after parsing anyway
      // and it gives correct invalidity errors without extra effort
      "imagesize" => {
        let dims = try!(n_words_to_or_fail::<u16>(&tokens, 1, 1, 2, true, &line_str, line_num));
        
        in_object = false;
        scene.image_size = if dims.len() == 1 {
          if dims[0] < MIN_IMG_DIM { (MIN_IMG_DIM, MIN_IMG_DIM) } else { (dims[0], dims[0]) }
        } else {
          (if dims[0] < MIN_IMG_DIM { MIN_IMG_DIM } else { dims[0] },
           if dims[1] < MIN_IMG_DIM { MIN_IMG_DIM } else { dims[1] })
        }
      },

      "background" => {        
        let cvals = try!(n_words_to_or_fail::<f32>(&tokens, 1, 3, 3, true, &line_str, line_num));

        in_object = false;
        scene.background = colour::from_slice(cvals.as_slice());
      },

      "ambient" => {
        let cvals = try!(n_words_to_or_fail::<f32>(&tokens, 1, 3, 3, true, &line_str, line_num));

        in_object = false;
        scene.ambient = colour::from_slice(cvals.as_slice());
      },

      "light" => {
        let p = try!(n_words_to_or_fail::<f64>(&tokens, 1, 3, 3, false, &line_str, line_num));
        let c = try!(n_words_to_or_fail::<f32>(&tokens, 4, 3, 3, true, &line_str, line_num));

        let light = Light { position: point!(p.as_slice()), 
                            colour: colour::from_slice(c.as_slice()),
        };
        in_object = false;
        scene.lights.push(light);
      },

      // TODO other shapes: bar, cylinder, triangle
      "sphere" => {
        let rval = try!(n_words_to_or_fail::<f64>(&tokens, 1, 1, 1, true, &line_str, line_num));
        let mut sphere = sphere!();


        let r = rval[0];
        if r > 0.0 {
          sphere.radius = r;
        } else {
          return Err(Error { kind: ErrorKind::InvalidLine,
                             desc: line_str.clone(),
                             line: LineNumber::Line(line_num),
          })
        }
        
        in_object = true;
        scene.spheres.push(sphere);
      },

      // TODO shouldn't really allow materials to be omitted, nor duplicate properties
      // Object properties
      "inner" if in_object => {
        let sphere = scene.spheres.last_mut().unwrap();

        let cols = try!(n_words_to_or_fail::<f32>(&tokens, 1, 9, 9, false, &line_str, line_num));
        let coeff = try!(n_words_to_or_fail::<u8>(&tokens, 10, 1, 1, true, &line_str, line_num));
        
        sphere.inner = new_material(cols[0..3], cols[3..6], cols[6..9], coeff[0]);
      },

      "outer" if in_object => {
        let sphere = scene.spheres.last_mut().unwrap();

        let cols = try!(n_words_to_or_fail::<f32>(&tokens, 1, 9, 9, false, &line_str, line_num));
        let coeff = try!(n_words_to_or_fail::<u8>(&tokens, 10, 1, 1, true, &line_str, line_num));
        
        sphere.outer = new_material(cols[0..3], cols[3..6], cols[6..9], coeff[0]);
      },

      "translate" if in_object => {
        let offsets = try!(n_words_to_or_fail::<f64>(&tokens, 1, 3, 3, true, &line_str, line_num));

        let this_t = mat4::new_translation(&vector!(offsets));
        let this_i = mat4::new_translation(&(vector!(offsets) * -1.0));

        let sphere = scene.spheres.last_mut().unwrap();
        // premultiply T and postmultiply Inv
        sphere.transform = mat4::multiply(&this_t, &sphere.transform);
        sphere.inverse_t = mat4::multiply(&sphere.inverse_t, &this_i);
      },

      "scale" if in_object => {
        let scalars = try!(n_words_to_or_fail::<f64>(&tokens, 1, 3, 3, true, &line_str, line_num));

        let this_t = mat4::new_scale(&vector!(scalars));
        let this_i = mat4::new_scale(&vector!(scalars).as_divisor_of(1.0));

        let sphere = scene.spheres.last_mut().unwrap();
        // premultiply T and postmultiply Inv
        sphere.transform = mat4::multiply(&this_t, &sphere.transform);
        sphere.inverse_t = mat4::multiply(&sphere.inverse_t, &this_i);
      },

      "rotate" if in_object => {
        
        let (axis_v, angle_d) = if n_toks == 5 {
          (try!(n_words_to_or_fail::<f64>(&tokens, 1, 3, 3, false, &line_str, line_num)),
           try!(n_words_to_or_fail::<i16>(&tokens, 4, 1, 1, true, &line_str, line_num))[0])
        } else if n_toks == 3 {
          (match tokens[1] {
            "X"|"x" => vec!(1.0, 0.0, 0.0),
            "Y"|"y" => vec!(0.0, 1.0, 0.0),
            "Z"|"z" => vec!(0.0, 0.0, 1.0),
            _ => return Err(Error { kind: ErrorKind::InvalidLine,
                                    desc: line_str.clone(),
                                    line: LineNumber::Line(line_num),
            })
          },
           try!(n_words_to_or_fail::<i16>(&tokens, 2, 1, 1, true, &line_str, line_num))[0])
        } else {
          return Err(Error { kind: ErrorKind::InvalidLine,
                             desc: line_str.clone(),
                             line: LineNumber::Line(line_num),
          })
        };
        
        let axis = vector!(axis_v);
        let angle_r = radians!(angle_d);

        let this_t = mat4::new_rotation(&axis, angle_r);
        let this_i = mat4::new_rotation(&axis, -1.0 * angle_r);

        let sphere = scene.spheres.last_mut().unwrap();
        // premultiply T and postmultiply Inv
        sphere.transform = mat4::multiply(&this_t, &sphere.transform);
        sphere.inverse_t = mat4::multiply(&sphere.inverse_t, &this_i);
        
      },

      _ => {
        return Err(Error { kind: ErrorKind::UnexpectedLine,
                           desc: line_str.clone(),
                           line: LineNumber::Line(line_num),
        })
      }
    }

    line_num += 1;
  }

  Ok(scene)
}

// Private functions
fn n_words_to_or_fail<T: FromStr>(items: &Vec<&str>, start: uint,
                                    min_length: uint, max_length: uint, expect_end: bool,
                                    line: &String, line_num: uint) -> ParseResult<Vec<T>> {
  assert!(min_length <= max_length);
  
  let working_len = if items.len() > start { items.len() - start } else { 0 };
  
  // if too few values or we expect the line to finish and we have too many then fail
  if min_length > working_len || (expect_end && working_len > max_length) {
    return Err(Error { kind: ErrorKind::InvalidLine,
                       desc: line.clone(),
                       line: LineNumber::Line(line_num),
    });
  }

  let result = items.iter()
    .skip(start)
    .map(|&s| from_str::<T>(s))
    .take(max_length)
    .take_while(|s| s.is_some())
    .filter_map(|s| s)
    .collect::<Vec<T>>();

  let length = result.len();

  // If we have extracted too few values
  if cmp::min(max_length, working_len) != length {
    return Err(Error { kind: ErrorKind::InvalidLine,
                       desc: line.clone(),
                       line: LineNumber::Line(line_num),
    });
  }

  Ok(result)
}
