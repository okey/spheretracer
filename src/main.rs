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
// todo fix namespace
use mat4::transpose;
use vec4::{Vec4,dot,add,sub,normalise,parametric_position,transform_multiply,as_vector,as_point,scale_by,magnitude};
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
  //while progress < colour_data.len() {
  //  progress = render_step(scene, colour_data, progress);
  //}
  println!("done");
  let mut chunk = 0;
  let chunk_size = wy * 4;
  let max_chunk = colour_data.len() / chunk_size;
  
  // TODO fix crap performance by using streaming, mapping the buffer, or using framebuffers etc
  // need to somehow decouple/desync changing the pixels and pushing to GPU
  while !window.should_close() {
    glfw.poll_events();
    
    while chunk <= max_chunk && progress < chunk_size * chunk {
      progress = render_step(scene, colour_data, progress);
    }
    chunk += 1;

    unsafe { 
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

type TestOpt = Option<(f64, f64)>;
type HitS<'a> = (&'a scene::Sphere, f64, f64);
type HitSOpt<'a> = Option<HitS<'a>>;

fn intersect_fixed_sphere(u: &Vec4, v: &Vec4, r: f64) -> TestOpt {
  // TODO perf these are the same for each object for a ray, so could manually cache them
  let uu = dot(u, u);
  let uv = dot(u, v);
  let vv = dot(v, v);
  
  let a = vv;
  let b = 2.0 * uv;
  let c = uu - r * r;

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

  match intersect_fixed_sphere(&uprime, &vprime, sphere.radius) {
    Some(t) => Some((sphere, t.val0(), t.val1())),
    _ => None
  }
}

fn trace_ray(scene: &scene::Scene, u: &Vec4, v: &Vec4) -> Colour {
  // TODO if recurse depth reached return BLACK

  // get sphere with max intersection less than (in front of) u.z
  // t will be negative if the hit is behind u

  // would be cleaner but less mathematically correct to return only nearest +ve t further down
  let hits: Vec<HitS> = scene.spheres.iter()
    .filter_map(|s| intersect_sphere(s, u, v)) // if we intersect
    .filter(|h| h.val1() >= 0.0 || h.val2() >= 0.0) // if we intersect within view
    .collect::<Vec<HitS>>();
  
  let fake_s = sphere!();
  let fake_h = (&fake_s, std::f64::MAX_VALUE, std::f64::MAX_VALUE);
  let hit = hits.iter()
    .fold(fake_h, |s, h| { if h.val1().min(h.val2()) < s.val1().min(s.val2()) { *h } else { s }});

  // double check my logic
  assert!(hits.len() != 0  || (hit.val1() == fake_h.val1() && hit.val2() == fake_h.val2()));
  
  if hits.len() == 0 { return scene.background }

  let t1 = hit.val1();
  let t2 = hit.val2();
  let sphere = &hit.val0();
  let t = if t1 >= 0.0 && t2 >= 0.0 { t1.min(t2) } else if t1 >= 0.0 { t1 } else { t2 };

  let hit_u = parametric_position(u, v, t);
  let hit_v = normalise(&sub(u, &hit_u)); // this is just unit(-v)
  let n_hat = normalise(&transform_multiply(&transpose(&sphere.inverse_t), &as_vector(&hit_u)));
  let v_hat = hit_v;//normalise(v); // not sure about this...

  let diffuse = &sphere.inner.diffuse;
  let phong = &sphere.inner.phong;
  // let ka = 1.0 therefore we do not scale the ambience further
  // scale colour by the intensity and colour of ambience
  let mut result_colour = colour::colour_multiply(diffuse, &scene.ambient);


  unsafe { if debug {
    println!("\nn={} hv={}", n_hat, hit_v);
  }}
  
  // TODO break out functions for diffuse and phong
  for light in scene.lights.iter() {
    // i_hat is a unit vector in direction of light source
    let i_vec = sub(&light.position, &hit_u);
    let i_hat = normalise(&i_vec);

    // TODO function
    // 0.0..1 is to fix cancer TODO apply when calculating u_hit
    let light_t = magnitude(&i_vec);
    let hits: Vec<HitS> = scene.spheres.iter()
      .filter_map(|s| intersect_sphere(s, &hit_u, &i_hat))
      .filter(|h|
              ((h.val1() >= 0.00001 && h.val1() < light_t) || 
               (h.val2() >= 0.00001 && h.val2() < light_t)))
      .collect::<Vec<HitS>>();

    unsafe { if debug {
      println!("hits:{}", hits);
    }}

    if hits.len() > 0 { continue; } // in shadow

    // Diffuse component
    // let kd = 1.0
    // therefore for each light add I[j]*n.i 
    
    let diff_light = colour::colour_scale(&light.colour, dot(&n_hat, &i_hat) as f32);
    // could extract the following multiplication outside the loop if I slackened colour clamping
    let diff_part = colour::colour_multiply(&diff_light, diffuse);
    result_colour = colour::colour_add(&result_colour, &diff_part);

    // Phong component
    // let kp = 1.0
    // therefore for each light add I[j]*(v.r)**n
  
    // r_hat is a unit vector of the light vector reflected about the normal
    // r_hat = i_hat - 2*(i_hat.n_hat)*n_hat
    let ni = scale_by(&n_hat, 2.0 * dot(&i_hat, &n_hat));
    let r_hat = sub(&i_hat, &ni);

    let rv = dot(&v_hat, &r_hat);
    let real_rv = if rv < 0.0 { 0.0 } else { rv };
    let phong_scale = std::num::pow(real_rv, sphere.inner.phong_n as uint);
    let phong_light = colour::colour_scale(&light.colour, phong_scale as f32);
    let phong_part  = colour::colour_multiply(&phong_light, phong); // could also be extracted
    result_colour = colour::colour_add(&result_colour, &phong_part);

    unsafe { if debug {
      println!("\nr={} ni{}", r_hat, ni);
    }}
  }

  // TODO soft shadows (light sampling) -  use random sampling
  // TODO reflection with bounded recursion
  // TODO transparency
  // TODO inner/outer hits and materials

  return result_colour;
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
  
  unsafe { debug = row == hx && col == hy; }
  let colour = trace_ray(scene, &u, &v);
  // TODO freeze scene after IO??
  colour_data[progress] = colour.red;
  colour_data[progress + 1] = colour.green;
  colour_data[progress + 2] = colour.blue;

  // TODO antialiasing

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
