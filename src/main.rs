#![feature(slicing_syntax)]
#![feature(macro_rules)]
#![feature(phase)]
#![feature(globs)]
#![feature(if_let)]

extern crate gl;
extern crate glfw;

use std::{os,mem,ptr,rand};
use std::path::Path;

use gl::types::*;
use glfw::Context;

use sceneio::read_scene;
// todo fix namespace
use mat4::transpose;
use vec4::{Vec4,DotProduct,Normalise,Transform,AsVector,Apply};
// TODO clean up trait usage once UFCS has been implemented in Rust
use colour::{Colour};
use scene::{Sphere,Material};

// module definition order matters for macro exports!
mod colour;
mod vec4;
mod mat4;
mod scene;
mod sceneio;

mod shaders;


const COLOUR_WIDTH:uint = 4;
const VERTEX_WIDTH:uint = 2;

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
type HitS<'a> = (&'a scene::Sphere, f64, Vec4, bool); // REALLY should be a struct at this point
type HitSOpt<'a> = Option<HitS<'a>>;

fn intersect_fixed_sphere(u: &Vec4, v: &Vec4, r: f64) -> TestOpt {
  // TODO perf these are the same for each object for a ray, so could manually cache them
  let uu = u.dot(u);
  let uv = u.dot(v);
  let vv = v.dot(v);
  
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
  let uprime = u.transform(&sphere.inverse_t);//transform_multiply(&sphere.inverse_t, u);
  let vprime = v.transform(&sphere.inverse_t);//transform_multiply(&sphere.inverse_t, v);

  match intersect_fixed_sphere(&uprime, &vprime, sphere.radius) {
    Some(x) => {
      let t1 = x.val0();
      let t2 = x.val1();
      
      // Consider the ray to have missed if an intersection is behind the ray's start position
      // Only return the nearest (non -ve) intersection with this object
      let (t, outside) = if t1 >= 0.0 && t2 >= 0.0 { (t1.min(t2), true) }
      else if t1 >= 0.0 { (t1, false) }
      else if t2 >= 0.0 { (t2, false) }
      else { return None };

      let h = vec4::parametric_position(&uprime, &vprime, t);
      
      Some((sphere, t, h, outside))
    },
    _ => None
  }
}

fn soft_shadow_scale(scene: &scene::Scene, light: &scene::Light, hit_u: &Vec4) -> f64 {
  const SOFT_SHADOW_SAMPLES: uint = 20;
  const SOFT_SHADOW_COEFF: f64 = 0.05;
  
  // TODO cache and pass ref to this
  let rng = rand::random::<f64>;

  // Jitter the light position by +/-0.5 * coeff in each axis to model lights with area
  let light_i_fn = if SOFT_SHADOW_SAMPLES == 1 {
    |p: &Vec4| *p - *hit_u // No jitter if we are only taking one sample (i.e. no soft shadows)
  } else {
    |p: &Vec4| p.apply(|c: f64| c + (rng() - 0.5) * SOFT_SHADOW_COEFF) - *hit_u
  };
  
  // Random sampling. Maybe a different distribution would look better?      
  let visible_samples = range(0, SOFT_SHADOW_SAMPLES).filter_map(|_| {
    let jit_i_vec = light_i_fn(&light.position);
    
    // If this sample is in shadow then do not count it
    let blocked = scene.spheres.iter().any(|s| {
      if let Some(h) = intersect_sphere(s, hit_u, &jit_i_vec) {
        return h.val1() >= 0.0 && h.val1() <= 1.0
      };
      false
    });
    
    if !blocked { return Some::<uint>(1) }
    None
  }).count();
  
  visible_samples as f64 / SOFT_SHADOW_SAMPLES as f64
}

// Phong illumination model
// hit_u is the hitpoint
// n_hat is the surface normal at the hitpoint
// v_hat is the vector from the hitpoint back to the viewpoint
fn illuminate_hit(scene: &scene::Scene, material: &scene::Material,
                  hit_u: &Vec4, n_hat: &Vec4, v_hat: &Vec4) -> Colour {
  // Ambient lighting
  // TODO no ambient component inside of a sphere unless it is transparent
  let mut result_colour = colour::colour_multiply(&material.diffuse, &scene.ambient);

  for light in scene.lights.iter() {
    // i_hat is a unit vector in direction of light source
    let i_hat = (light.position - *hit_u).normalise();
    
    let soften_scale = soft_shadow_scale(scene, light, hit_u); 
    if soften_scale == 0.0 { continue; } // in shadow

    // Diffuse (Lambertian) component
    let ni = n_hat.dot(&i_hat) * soften_scale;
    if ni > 0.0 { 
      let diff_light = colour::colour_scale(&light.colour, ni as f32);
      // could extract the following multiplication outside the loop 
      let diff_part = colour::colour_multiply(&diff_light, &material.diffuse);
      result_colour = colour::colour_add(&result_colour, &diff_part);
    }

    // Phong highlight component
    // r_hat is a unit vector of the light vector reflected about the normal
    let n_perp2 = *n_hat * (2.0 * i_hat.dot(n_hat));
    let r_hat = (n_perp2 - i_hat).normalise();
    // TODO extract vector reflection outside of lights loop by changing which vector we reflect

    let rv = v_hat.dot(&r_hat) * soften_scale;
    if rv > 0.0 {
      let phong_scale = std::num::pow(rv, material.phong_n as uint);
      let phong_light = colour::colour_scale(&light.colour, phong_scale as f32);
       // could also be extracted
      let phong_part  = colour::colour_multiply(&phong_light, &material.phong);
      result_colour = colour::colour_add(&result_colour, &phong_part);
    }
  }

  result_colour
}

fn trace_ray(scene: &scene::Scene, u: &Vec4, v: &Vec4, depth: uint) -> Colour {
  // Stop mirror ray recursion
  if depth == 0 { return colour::BLACK; }

  // Find all objects that lie in the path of the ray for non -ve t
  let hits = scene.spheres.iter()
    .filter_map(|s| intersect_sphere(s, u, v))
    .collect::<Vec<HitS>>();
  
  // Could multiply this with the ambient if desired
  if hits.len() == 0 { return scene.background; }
  
  
  // Take the closest hit and extract the result
  let fake_s = sphere!();
  let fake_h = (&fake_s, std::f64::MAX_VALUE, vector!(), false);
  let hit = hits.iter().fold(fake_h, |s, h| { if h.val1() < s.val1() { *h } else { s }});

  let sphere = &hit.val0();
  let material = if hit.val3() { sphere.outer } else { sphere.inner };
  
  // Find the hit position, surface normal at the hit position, and reverse viewpoint vectors
  let hit_u = vec4::parametric_position(u, v, hit.val1() * 0.9999);
  let n_hat = hit.val2().as_vector().transform(&mat4::transpose(&sphere.inverse_t)).normalise();
  let v_hat = (*u - hit_u).normalise();

  // Do illumination using the Phong model
  let phong_colour = illuminate_hit(scene, &material, &hit_u, &n_hat, &v_hat);
  
  // Trace any reflections
  let mirror_colour = if !colour::colour_eq(&colour::BLACK, &material.mirror) {
    let reflected_v = (n_hat * (2.0 * v_hat.dot(&n_hat)) - v_hat).normalise();
    let reflected_c = trace_ray(scene, &hit_u, &reflected_v, depth - 1);

    colour::colour_multiply(&material.mirror, &reflected_c)
  } else {
    colour::BLACK
  };
  let result_colour = colour::colour_add(&phong_colour, &mirror_colour);


  // TODO transparency

  return colour::colour_clamp(&result_colour);
}

// scene space => 1 ray per pixel(vertex), right handed system, but with 0,0 in the centre
// so need to translate by w/2 and h/2, don't flip Z because we should already be in an RH system
// and scale back to 2x2 plane
fn render_step(scene: &scene::Scene, colour_data: &mut [GLfloat], progress: uint) -> uint {
  let depth_max = 5;
  let wx = scene.image_size.val0() as int;
  let wy = scene.image_size.val1() as int;

  let row = progress as int / (COLOUR_WIDTH as int * wy);
  let col = (progress as int % (COLOUR_WIDTH as int * wy)) / COLOUR_WIDTH as int;

  let hx = wx / 2;
  let hy = wy / 2;

  // ray start
  // this is our viewpoint
  let u = point!(0.0 0.0 1.0);
  
  
  // use find the amount the direction changes per pixel
  // use the y-axis to fix the FOV so that 800x600 looks like 600x600 but showing more of the scene
  // instead of stretching the contents
  // Let our screen in object space be -1*wx/wy,-1,1 to 1*wx/wy,1,1
  // TODO this is always the same, could cache it
  let step_y = 1.0 / hy as f64;
  let step_half = step_y / 2.0;
  
  // ray direction
  // we add half the step so that the ray is in the centre of the sample rectangle
  let dx = (row as f64 - hx as f64) * step_y + step_half;
  let dy = (col as f64 - hy as f64) * step_y + step_half;
  let dz = -1.0; // cleaner than using 0.0-1.0 to get macro to work
  let v = vector!(dx dy dz);
  
  unsafe { debug = row == hx && col == hy; } // track the centre ray for debugging
  let colour = trace_ray(scene, &u, &v, depth_max);

  // TODO check scene is correctly frozen after IO
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
