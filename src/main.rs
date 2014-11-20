#![feature(slicing_syntax)]
#![feature(macro_rules)]
#![feature(phase)]
#![feature(globs)]

extern crate gl;
extern crate glfw;

use std::{os,mem,ptr};
use std::path::Path;

use gl::types::*;
use glfw::Context;

use sceneio::read_scene;
use vec4::{Vec4,dot,transform_multiply};
use colour::Colour;
use scene::{Sphere,Material};

// module definition order matters for macro exports!
mod colour;
mod vec4;
mod mat4;
mod scene;
mod sceneio;

mod shaders;


static COLOUR_WIDTH:uint = 4;
static VERTEX_WIDTH:uint = 2;

fn make_gl_proj_mat(width: uint, height: uint) -> [GLfloat, ..16] {
  let mut m:[GLfloat, ..16] = [0.0, ..16];
  let w = width as f32;
  let h = height as f32;
  let f = 0.1;
  let n = 0.0;

  // from NDC to world space
  // translate by -1,-1 to put origin in bottom left
  // scale to change square 2,2 to rectangle w,h
  // flip Z because NDC is left handed

  // OpenGL matrix memory layout has transposition vector as the last row like D3D
  m[0] = 2.0 / w; // 2/(r-l)
  m[12] = -1.0;   // direct construction, so -1.0 rather than -(r+l)/(r-l)
  
  m[5] = 2.0 / h; // 2/(t-b)
  m[13] = -1.0;   // direct construction, so -1.0 rather than -(t+b)/(t-l)

  m[10] = -2.0 / (f - n);
  m[14] = -1.0 * (f + n) / (f - n);

  m[15] = 1.0;
  
  m
}

fn gl_init_and_render(scene: &scene::Scene) {
  let wx = scene.image_size.val0() as uint;
  let wy = scene.image_size.val1() as uint;

  let glfw = glfw::init(glfw::FAIL_ON_ERRORS).unwrap();

  // Choose a GL profile that is compatible with OS X 10.7+
  glfw.window_hint(glfw::ContextVersion(3, 2));
  glfw.window_hint(glfw::OpenglForwardCompat(true));
  glfw.window_hint(glfw::OpenglProfile(glfw::OpenGlCoreProfile));
  glfw.window_hint(glfw::Resizable(false));

  let (window, _) = glfw.create_window(wx as u32, wy as u32, "OpenGL", glfw::Windowed)
    .expect("Failed to create GLFW window.");

  // It is essential to make the context current before calling `gl::load_with`.
  window.make_current();

  // Load the OpenGL function pointers
  gl::load_with(|s| window.get_proc_address(s));

  // Create GLSL shaders
  let vs = shaders::compile_shader(shaders::VS_SRC, gl::VERTEX_SHADER);
  let fs = shaders::compile_shader(shaders::FS_SRC, gl::FRAGMENT_SHADER);
  let program = shaders::link_program(vs, fs);
  let mut vao = 0;
  
  // Add a vertex at every pixel... seems to cause issues at any res other than 800x600
  // TODO try building a texture instead
  let mut vbo = 0;
  let wy2 = wy * VERTEX_WIDTH;
  let vertex_data_vec: Vec<GLfloat> = Vec::from_fn(wx * wy2, |n| {
    let x = n / wy2;
    let y = n % wy2;
    let v = if y % 2 == 0 { x as f32 } else { (y / VERTEX_WIDTH) as f32};
    v + 0.5
  });
  let vertex_data = vertex_data_vec.as_slice();
  println!("{}", vertex_data[wx * wy2 - 4..]);
  
  // Add a colour for each vertex
  let mut cbo = 0;
  let wy4 = wy * COLOUR_WIDTH;
  let mut colour_data_vec: Vec<GLfloat> = Vec::from_fn(wx * wy4, |n| {
    match n % COLOUR_WIDTH {
      0 => (scene.background.red as f32   / 255.0) as GLfloat,
      1 => (scene.background.green as f32 / 255.0) as GLfloat,
      2 => (scene.background.blue as f32  / 255.0) as GLfloat,
      _ => 1.0
    }
  });
  let colour_data = colour_data_vec.as_mut_slice();

  unsafe {
    // Create Vertex Array Object
    gl::GenVertexArrays(1, &mut vao);
    gl::BindVertexArray(vao);

    // Set up vertex buffer object
    gl::GenBuffers(1, &mut vbo);
    gl::BindBuffer(gl::ARRAY_BUFFER, vbo);
    gl::BufferData(gl::ARRAY_BUFFER,
                   (vertex_data.len() * mem::size_of::<GLfloat>()) as GLsizeiptr, 
                   mem::transmute(&vertex_data[0]),
                   gl::STATIC_DRAW);

    // Set up colour buffer object
    gl::GenBuffers(1, &mut cbo);
    gl::BindBuffer(gl::ARRAY_BUFFER, cbo);
    gl::BufferData(gl::ARRAY_BUFFER,
                   (colour_data.len() * mem::size_of::<GLfloat>()) as GLsizeiptr, 
                   mem::transmute(&colour_data[0]),
                   gl::STATIC_DRAW);

    gl::UseProgram(program);
    // Bind fragment shader
    "out_colour".with_c_str(|ptr| gl::BindFragDataLocation(program, 0, ptr));

    // Configure vertex buffer
    let pos_attr = "position".with_c_str(|ptr| gl::GetAttribLocation(program, ptr));
    gl::EnableVertexAttribArray(pos_attr as GLuint);
    gl::BindBuffer(gl::ARRAY_BUFFER, vbo);
    gl::VertexAttribPointer(pos_attr as GLuint, VERTEX_WIDTH as i32, gl::FLOAT,
                            gl::FALSE as GLboolean, 0, ptr::null());


    // Set up the projection manually because we don't have glm or glu
    let proj = make_gl_proj_mat(wx, wy);
    let mvp_uni = "mvp".with_c_str(|ptr| gl::GetUniformLocation(program, ptr));
    
    // 3rd argument is transpose. GLES does not support this
    gl::UniformMatrix4fv(mvp_uni as GLint, 1, gl::FALSE as GLboolean,
                         mem::transmute(&proj[0]));
   
    // Configure colour buffer
    let col_attr = "vertex_colour".with_c_str(|ptr| gl::GetAttribLocation(program, ptr));
    gl::EnableVertexAttribArray(col_attr as GLuint);
    gl::BindBuffer(gl::ARRAY_BUFFER, cbo);
    gl::VertexAttribPointer(col_attr as GLuint, COLOUR_WIDTH as i32, gl::FLOAT,
                            gl::FALSE as GLboolean, 0, ptr::null());
  }

  let mut progress = 0;
  println!("starting loop");
  while progress < colour_data.len() {
    progress = render_step(scene, colour_data, progress);
  }
  println!("done");
  
  // TODO fix crap performance by using streaming, mapping the buffer, or using framebuffers etc
  // need to somehow decouple/desync changing the pixels and pushing to GPU
  while !window.should_close() {
    glfw.poll_events();
    unsafe {

      /*if progress < colour_data.len() {
        progress = render(scene, colour_data, progress);
      }*/
      
      gl::BufferData(gl::ARRAY_BUFFER,
                     (colour_data.len() * mem::size_of::<GLfloat>()) as GLsizeiptr, 
                     mem::transmute(&colour_data[0]),
                     gl::STATIC_DRAW);
      
      
      gl::ClearColor(0.3, 0.3, 0.3, 1.0);
      gl::Clear(gl::COLOR_BUFFER_BIT);
      gl::DrawArrays(gl::POINTS, 0, vertex_data.len() as i32);
    }
    window.swap_buffers();
  }

  unsafe {
    gl::DeleteProgram(program);
    gl::DeleteShader(fs);
    gl::DeleteShader(vs);
    gl::DeleteBuffers(1, &cbo);
    gl::DeleteBuffers(1, &vbo);
    gl::DeleteVertexArrays(1, &vao);
  }
}

//fn trace(scene: &scene::Scene, ) -> Colour {
// 
//}

type TestOpt = Option<(f64, f64)>;
type HitS<'a> = (&'a scene::Sphere, f64, f64);
type HitSOpt<'a> = Option<HitS<'a>>;

fn intersect_unit_sphere(u: &Vec4, v: &Vec4) -> TestOpt {
  // TODO perf these are the same for each object for a ray, so could manually cache them
  let uu = dot(u, u);
  let uv = dot(u, v);
  let vv = dot(v, v);
  
  let a = vv;
  let b = 2.0 * uv;
  let c = uu - 1.0;

  let ac4 = a * c * 4.0;
  let bsq =  b * b;
  
  if bsq <= ac4 {
    // < no real roots; miss
    // = 1  real root ; ray is tangent to sphere; treat as miss
    return None
  }

  let p = (bsq -  ac4).sqrt();
  
  let t1 = if b > 0.0 { (-b - p) / (2.0 * a) } else { (-b + p) / (2.0 * a) };
  let t2 = c / (a * t1);
  
  Some((t1, t2))
}

static mut debug: bool = false;
fn intersect_sphere<'a>(sphere: &'a scene::Sphere,  u: &Vec4, v: &Vec4) -> HitSOpt<'a> {
  let uprime = transform_multiply(&sphere.inverse_t, u);
  let vprime = transform_multiply(&sphere.inverse_t, v);

  unsafe { if debug {
    println!("s{} {}", u, v);
    println!("'{} {}", uprime, vprime);
  }}

  match intersect_unit_sphere(&uprime, &vprime) {
    Some(t) => Some((sphere, t.val0(), t.val1())),
    _ => None
  }
}

// TODO recursion depth
fn trace_ray(scene: &scene::Scene, u: &Vec4, v: &Vec4) -> Colour {
  // get sphere with max intersection less than (in front of) u.z
  // t will be negative if the hit is behind u

  // would be cleaner but less mathematically correct to return only nearest +ve t further down
  let hits: Vec<HitS> = scene.spheres.iter()
    .filter_map(|s| intersect_sphere(s, u, v))
    .filter(|h| h.val1() >= 0.0 || h.val2() >= 0.0)
    .collect::<Vec<HitS>>();
  
  let fake_s = sphere!();
  let fake_h = (&fake_s, std::f64::MAX_VALUE, std::f64::MAX_VALUE);
  let hit = hits.iter()
    .fold(fake_h, |s, h| { if h.val1().min(h.val2()) < s.val1().min(s.val2()) { *h } else { s }});

  if hit.val1() == fake_h.val1() && hit.val2() == fake_h.val2() {
    return scene.background;
    //return colour::BLACK;
  }

  return hit.val0().inner.diffuse;
}

// scene space => 1 ray per pixel(vertex), right handed system, but with 0,0 in the centre
// so need to translate by w/2 and h/2, don't flip Z because we should already be in an RH system
// and scale back to 2x2 plane
fn render_step(scene: &scene::Scene, colour_data: &mut [GLfloat], progress: uint) -> uint {
  let wx = scene.image_size.val0() as int;
  let wy = scene.image_size.val1() as int;

  let row = progress as int / (COLOUR_WIDTH as int * wy);
  let col = (progress as int % (COLOUR_WIDTH as int * wy)) / COLOUR_WIDTH as int;

  let hx = wx / 2;
  let hy = wy / 2;

  // Let our screen in object space be -1*wx/wy,-1,1 to 1*wx/wy,1,1
  // ray start
  let px = ((row - hx) as f64 + 0.5) / hy as f64;
  let py = ((col - hy) as f64 + 0.5) / hy as f64;
  let u = point!(px py 1.0);

  // ray direction
  // this is always the same, could cache it
  let dz = -1.0; // cleaner than using 0.0-1.0 to get macro to work
  let v = vector!(0.0 0.0 dz);

  /*let result = intersect_unit_sphere(&u, &v);

  if result != None {
    colour_data[progress] = 1.0;
    colour_data[progress + 2] = 0.0;
  }*/

  
  unsafe { debug = row == 400 && col == 300; }
  let colour = trace_ray(scene, &u, &v);
  colour_data[progress] = colour.red as f32 / 255.0;
  colour_data[progress + 1] = colour.green as f32 / 255.0;
  colour_data[progress + 2] = colour.blue as f32 / 255.0;

  progress + 4
}

fn load_scene_or_fail(filename: &Path) -> scene::Scene {
  let scene = match read_scene(filename) {
    Ok(scene) => {
      println!("\nParsed {}: {}", filename.display(), scene);
      scene
    },
    Err(desc) => {
      panic!("\nParsing {} failed\n\t {}", filename.display(), desc);
    }
  };

  for s in scene.spheres.iter() {
    mat4::matrix_print(&s.transform, "Transform");
    mat4::matrix_print(&s.inverse_t, "Inverse T");
  }
  
  scene
}

fn main() {
  let args = os::args();
  let print_usage = args.len() != 2u;

  if print_usage {
    println!("Usage:\n\t {} scene_file.txt", args[0]);
    return
  }
  
  let filename = Path::new(args[1].clone());
  let scene = load_scene_or_fail(&filename);

  gl_init_and_render(&scene);
}
