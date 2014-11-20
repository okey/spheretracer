use std::f64::consts as f64consts;
use std::error::FromError;
use std::io::{File,BufferedReader,IoError};
use std::path::Path;
use std::fmt;

use mat4;
use vec4;
use vec4::Vec4;
use colour;
use colour::Colour;
use scene::{Scene,Light,Material,Sphere,make_colour,make_material};

/// markdown
// comment

#[deriving(Show)]
enum ParseErrorKind {
  FileOperationFailed,
  UnexpectedLine,
  InvalidLine,
}

// possibly not worthwhile
#[deriving(Show)]
enum LineNumber<T> {
  Line(T),
  NoLine,
}

type ULineNumber = LineNumber<uint>;

pub struct ParseError {
  pub kind: ParseErrorKind,
  pub desc: String, // can't use &'static str if I want to pass in the contents of the failed line
  pub line: ULineNumber,
}

impl FromError<IoError> for ParseError {
  fn from_error(e: IoError) -> ParseError {
    match e {
      _ => ParseError { kind: FileOperationFailed, desc: String::from_str(e.desc), line: NoLine },
    }
  }
}

impl fmt::Show for ParseError {
  fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
    match self.kind {
      FileOperationFailed => write!(f, "{} : {}", self.kind, self.desc),
      _ => write!(f, "{} {}: {}", self.kind, self.line, self.desc)
    }
  }
}

type ParseResult = Result<Scene, ParseError>;


macro_rules! radians(
  ($degrees:expr) => {$degrees as f64 * f64consts::PI / 180.0};)


// Just using TOML and converting any old scenes with python would have been cleaner and easier, but I'm trying to learn Rust
pub fn read_scene(filename: &Path) -> ParseResult {
  let mut file = BufferedReader::new(File::open(filename));
  let mut line_num = 0u;

  // Default scene
  // TODO use defaults properly
  let mut scene = Scene {
    image_size: (1, 2),
    ambient: colour::WHITE,
    background: colour::BLACK,
    lights: Vec::new(),
    spheres: Vec::new()
  };

  // Used to reject attempts to set object properties from outside an object
  let mut in_object = false;
  // maybe there's an equivalent of .next() that would let me avoid this and use an inner consuming loop?
  // try advance()?

  for line in file.lines() {
    let ls = try!(line);

    // Can we do this more cleanly and without building a vector?
    let mut tokens: Vec<&str> = Vec::new();
    for tok in ls.trim().split_str(" ") { // TODO handle \t
      if tok.starts_with("#") {
        break;
      } else if !tok.is_empty() {
        tokens.push(tok);
      }
    }

    // Skip blank lines and comments
    let n = tokens.len();
    if n == 0 {
      line_num += 1;
      continue;
    }


    let s = tokens[0];
    match s {
      // TODO abstract colour type a bit
      // Scene properties
      // Not using guards since we check the result after parsing anyway?
      // And it gives correct invalidity errors without extra effort
      "imagesize" => {
        in_object = false;
        let dims: Vec<u16> = tokens.tail().iter().filter_map(|&s| from_str::<u16>(s)).collect();
        let dslice = dims.as_slice();

        scene.image_size = match dims.len() {
          1 => (dslice[0], dslice[0]),
          2 => (dslice[0], dslice[1]),
          _ => return Err(ParseError { kind: InvalidLine, desc: ls.clone(), line: Line(line_num) })
        }
      },

      "background" => {
        in_object = false;
        let vals: Vec<f32> = tokens.tail().iter().filter_map(|&s| from_str::<f32>(s)).collect();

        scene.background = match vals.len() {
          3 => make_colour(vals.as_slice()),
          _ => return Err(ParseError { kind: InvalidLine, desc: ls.clone(), line: Line(line_num) })
        }
      },

      "ambient" => {
        in_object = false;
        let vals: Vec<f32> = tokens.tail().iter().filter_map(|&s| from_str::<f32>(s)).collect();

        scene.ambient = match vals.len() {
          3 => make_colour(vals.as_slice()),
          _ => return Err(ParseError { kind: InvalidLine, desc: ls.clone(), line: Line(line_num) })
        }
      },

      "light" => {
        in_object = false;
        let p: Vec<f64> = tokens.tail().iter()
          .take(3).filter_map(|&s| from_str::<f64>(s)).collect();
        let c: Vec<f32> = tokens.tail().iter()
          .skip(3).filter_map(|&s| from_str::<f32>(s)).collect();

        match p.len() + c.len() {
          6 => {
            let pslice = p.as_slice();
            let cslice = c.as_slice();
            let light = Light { position: point!(pslice), colour: make_colour(cslice) };
            scene.lights.push(light);

          },
          _ => return Err(ParseError { kind: InvalidLine, desc: ls.clone(), line: Line(line_num) })
        }
      },

      // TODO bar, cylinder, also tri?
      "sphere" => {
        in_object = true;
        let vals: Vec<f64> = tokens.tail().iter().filter_map(|&s| from_str::<f64>(s)).collect();
        let mut sphere = sphere!();

        match vals.len() {
          1 => {
            let r = vals.as_slice()[0];
            if r > 0.0 {
              sphere.radius = r;
            } else {
              return Err(ParseError { kind: InvalidLine, desc: ls.clone(), line: Line(line_num) })
            }
            scene.spheres.push(sphere);
          },
          _ => return Err(ParseError { kind: InvalidLine, desc: ls.clone(), line: Line(line_num) })
        }
      },

      // TODO shouldn't really allow materials to be omitted
      // Object properties
      "inner"     if in_object && n == 11 => {
        // slice_or_fail(&1u, &10u) is pretty ugly but more stable?
        let colours: Vec<f32> = tokens[1..10].iter().filter_map(|&s| from_str::<f32>(s)).collect();
        let phong_n =  match from_str::<u8>(tokens[10]) {
          Some(val) => val,
          _ => return Err(ParseError { kind: InvalidLine, desc: ls.clone(),
                                       line: Line(line_num) })
        };

        let sphere = match scene.spheres.last_mut() {
          Some(val) => val,
          _ => return Err(ParseError { kind: UnexpectedLine, desc: ls.clone(),
                                       line: Line(line_num) })
        };

        if colours.len() == 9 {
          sphere.inner = make_material(colours[0..3], colours[3..6], colours[6..9], phong_n);
        } else {
          return Err(ParseError { kind: InvalidLine, desc: ls.clone(),
                                  line: Line(line_num) })
        }

      },

      // TODO put in a function and use try! ?
      "outer"     if in_object && n == 11 => {
        let colours: Vec<f32> = tokens[1..10].iter().filter_map(|&s| from_str::<f32>(s)).collect();
        let phong_n =  match from_str::<u8>(tokens[10]) {
          Some(val) => val,
          _ => return Err(ParseError { kind: InvalidLine, desc: ls.clone(),
                                       line: Line(line_num) })
        };

        let sphere = match scene.spheres.last_mut() {
          Some(val) => val,
          _ => return Err(ParseError { kind: UnexpectedLine,desc: ls.clone(),
                                       line: Line(line_num) })
        };

        if colours.len() == 9 {
          sphere.outer = make_material(colours[0..3], colours[3..6], colours[6..9], phong_n);
        } else {
          return Err(ParseError { kind: InvalidLine, desc: ls.clone(),
                                  line: Line(line_num) })
        }
      },

      "translate" if in_object && n == 4  => {
        let offsets: Vec<f64> = tokens.tail().iter().filter_map(|&s| from_str::<f64>(s)).collect();

        let sphere = match scene.spheres.last_mut() {
          Some(val) => val,
          _ => return Err(ParseError { kind: UnexpectedLine, desc: ls.clone(),
                                       line: Line(line_num) })
        };

        if offsets.len() == 3 {
          let this_t = mat4::translate(&vector!(offsets));
          let this_i = mat4::translate(&vec4::scale_by(&vector!(offsets), -1.0));

          // premultiply T and postmultiply I
          sphere.transform = mat4::multiply(&this_t, &sphere.transform);
          sphere.inverse_t = mat4::multiply(&sphere.inverse_t, &this_i);
        } else {
          return Err(ParseError { kind: InvalidLine, desc: ls.clone(),
                                  line: Line(line_num) })
        }
      },

      "scale"     if in_object && n == 4  => {
        let offsets: Vec<f64> = tokens.tail().iter().filter_map(|&s| from_str::<f64>(s)).collect();

        let sphere = match scene.spheres.last_mut() {
          Some(val) => val,
          _ => return Err(ParseError { kind: UnexpectedLine, desc: ls.clone(),
                                       line: Line(line_num) })
        };

        if offsets.len() == 3 {
          let this_t = mat4::scale(&vector!(offsets));
          let this_i = mat4::scale(&vec4::as_divisor(1.0, &vector!(offsets)));

          // premultiply T and postmultiply I
          sphere.transform = mat4::multiply(&this_t, &sphere.transform);
          sphere.inverse_t = mat4::multiply(&sphere.inverse_t, &this_i);
        } else {
          return Err(ParseError { kind: InvalidLine, desc: ls.clone(),
                                  line: Line(line_num) })
        }
      },

      // TODO wouldn't another repr e.g. general axis angle vector be much less annoying?
      // even if the formula for the matrix is a little more involved, it's not that bad...
      "rotate"    if in_object && n == 3  => {
        let sphere = match scene.spheres.last_mut() {
          Some(val) => val,
          _ => return Err(ParseError { kind: UnexpectedLine, desc: ls.clone(),
                                       line: Line(line_num) })
        };

        let angle_r = match from_str::<u16>(tokens[2]) { // Allow only integer degrees for now
          Some(val) => radians!(val),
          _ => return Err(ParseError { kind: InvalidLine, desc: ls.clone(),
                                       line: Line(line_num) })
        };

        let axis = match tokens[1] {
          "X"|"x" => vector!(1.0 0.0 0.0),
          "Y"|"y" => vector!(0.0 1.0 0.0),
          "Z"|"z" => vector!(0.0 0.0 1.0),
          _ => return Err(ParseError { kind: InvalidLine, desc: ls.clone(),
                                       line: Line(line_num) })
        };

        let this_t = mat4::rotate(&axis, angle_r);
        let this_i = mat4::rotate(&axis, -1.0 * angle_r);


        // premultiply T and postmultiply I
        sphere.transform = mat4::multiply(&this_t, &sphere.transform);
        sphere.inverse_t = mat4::multiply(&sphere.inverse_t, &this_i);
      },

      _ => {
        return Err(ParseError { kind: UnexpectedLine, desc: ls.clone(),
                                line: Line(line_num) })
      }
    }

    line_num += 1;
  }

  Ok(scene)
}
