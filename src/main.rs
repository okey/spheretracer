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
use mat4::transpose;
use vec3::{Vec3,DotProduct,Normalise,Transform,AsVector,Apply};
// TODO clean up trait usage once UFCS has been implemented in Rust
use colour::{Colour,Clamp};
use scene::{Sphere,Material};

// NOTE module definition order matters for macro exports!
pub mod colour;
pub mod vec3;
pub mod mat4;
pub mod scene;
mod sceneio;

pub mod shaders;

/* The number of components expected per value (colour or position) in the data arrays drawn */
const COLOUR_WIDTH:uint = 4;
const VERTEX_WIDTH:uint = 2;


/* Make an orthographic matrix to transform from NDC to world (scene) space */
fn make_gl_ortho_mat(width: uint, height: uint) -> [GLfloat, ..16] {
  let mut m:[GLfloat, ..16] = [0.0, ..16];
  let w = width as f32;
  let h = height as f32;
  let f = 0.1;
  let n = 0.0;

  // from NDC to world space:
  // translate by -1,-1 to put origin in bottom left
  // scale to change square 2,2 to rectangle w,h
  // flip Z because NDC is left handed
  // set the near and far planes to draw z=[0.0,0.1)

  // OpenGL matrix memory layout has transposition vector as the last row like D3D
  m[0] = 2.0 / w;
  m[12] = -1.0;
  
  m[5] = 2.0 / h;
  m[13] = -1.0;

  m[10] = -2.0 / (f - n);
  m[14] = -1.0 * (f + n) / (f - n);

  m[15] = 1.0;
  
  m
}

fn gl_init_and_render(scene: &scene::Scene) {
  let wx = scene.image_size.val0() as uint;
  let wy = scene.image_size.val1() as uint;

  let glfw = glfw::init(glfw::FAIL_ON_ERRORS).unwrap();

  // Select GL profile; this one should be cross-platform
  glfw.window_hint(glfw::ContextVersion(3, 2));
  glfw.window_hint(glfw::OpenglForwardCompat(true));
  glfw.window_hint(glfw::OpenglProfile(glfw::OpenGlCoreProfile));
  glfw.window_hint(glfw::Resizable(false));

  let (window, _) = glfw.create_window(wx as u32, wy as u32, "OpenGL", glfw::Windowed)
    .expect("Failed to create GLFW window.");

  // Must do this before loading function pointers
  window.make_current();

  // Load the OpenGL function pointers
  gl::load_with(|s| window.get_proc_address(s));

  // Create GLSL shaders
  let vs = shaders::compile_shader(shaders::VS_SRC, gl::VERTEX_SHADER);
  let fs = shaders::compile_shader(shaders::FS_SRC, gl::FRAGMENT_SHADER);
  let program = shaders::link_program(vs, fs);
  let mut vao = 0;
  
  // Add a vertex at every pixel... simple but not idiomatic OpenGL
  let mut vbo = 0;
  let wy2 = wy * VERTEX_WIDTH;
  let vertex_data_vec: Vec<GLfloat> = Vec::from_fn(wx * wy2, |n| {
    let x = n / wy2;
    let y = n % wy2;
    let v = if y % 2 == 0 { x as f32 } else { (y / VERTEX_WIDTH) as f32};
    v + 0.5
  });
  let vertex_data = vertex_data_vec.as_slice();
  
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

    // Select the program
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
    let proj = make_gl_ortho_mat(wx, wy);
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

  // Do the raytracing in chunks so we can watch it happen on screen
  let mut chunk = 0;
  let chunk_size = wy * 4;
  let max_chunk = colour_data.len() / chunk_size;
  
  let mut render_progress = 0;
  // TODO better GL performance for scenes that render quickly
  // Streaming, mapping the buffer, or writing to a dynamic texture are a few options
  while !window.should_close() {
    glfw.poll_events();
    
    while chunk <= max_chunk && render_progress < chunk_size * chunk {
      render_progress = render_step(scene, colour_data, render_progress);
    }
    chunk += 1;
    
    // cbo is still bound from setup
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
type HitS<'a> = (&'a scene::Sphere, f64, Vec3, bool); // REALLY should be a struct at this point
type HitSOpt<'a> = Option<HitS<'a>>;

fn intersect_fixed_sphere(u: &Vec3, v: &Vec3, r: f64) -> TestOpt {
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

fn intersect_sphere<'a>(sphere: &'a scene::Sphere,  u: &Vec3, v: &Vec3) -> HitSOpt<'a> {
  let uprime = u.transform(&sphere.inverse_t);
  let vprime = v.transform(&sphere.inverse_t);

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

      let h = vec3::parametric_position(&uprime, &vprime, t);
      
      Some((sphere, t, h, outside))
    },
    _ => None
  }
}

/* Calculate what factor to colour a position by, using rnadom smapling to soften shadows */
fn soft_shadow_scale(scene: &scene::Scene, light: &scene::Light, hit_u: &Vec3) -> f64 {
  const SOFT_SHADOW_SAMPLES: uint = 1;
  const SOFT_SHADOW_COEFF: f64 = 0.05;
  
  let rng = rand::random::<f64>; // TODO investigate performance of doing this here

  // Jitter the light position by +/-0.5 * coeff in each axis to model lights with area
  let light_i_fn = if SOFT_SHADOW_SAMPLES == 1 {
    |p: &Vec3| *p - *hit_u // No jitter if we are only taking one sample (i.e. no soft shadows)
  } else {
    |p: &Vec3| p.apply(|c: f64| c + (rng() - 0.5) * SOFT_SHADOW_COEFF) - *hit_u
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
  
  // Return the proprotion by which we should colour the hit position
  visible_samples as f64 / SOFT_SHADOW_SAMPLES as f64
}

/* Illuminate a hit using the Phong illumination model
 *
 * hit_u is the hit position
 * n_hat is the (unit) surface normal at the hit position
 * v_hat is the (unit) vector from the hit position back to the viewpoint
 */
fn illuminate_hit(scene: &scene::Scene, material: &scene::Material,
                  hit_u: &Vec3, n_hat: &Vec3, v_hat: &Vec3) -> Colour {
  // Ambient lighting
  // TODO no ambient component inside of a sphere unless it is transparent
  let mut result_colour: colour::Colour = material.diffuse * scene.ambient;

  // TODO extract vector reflection outside of lights loop by changing which vector gets reflected
  for light in scene.lights.iter() {
    // i_hat is a unit vector in direction of light source
    let i_hat = (light.position - *hit_u).normalise();
    
    let soften_scale = soft_shadow_scale(scene, light, hit_u); 
    if soften_scale == 0.0 { continue; } // in shadow

    // Diffuse (Lambertian) component
    let ni = n_hat.dot(&i_hat) * soften_scale;
    if ni > 0.0 { 
      let diff_light = light.colour * ni as f32;
      let diff_part = diff_light * material.diffuse;
      result_colour = result_colour + diff_part;
    }

    // Phong highlight component
    // r_hat is a unit vector of the light vector reflected about the normal
    let n_p_scale = *n_hat * (2.0 * i_hat.dot(n_hat));
    let r_hat = (n_p_scale - i_hat).normalise();

    let rv = v_hat.dot(&r_hat) * soften_scale;
    if rv > 0.0 {
      let phong_scale = std::num::pow(rv, material.phong_n as uint);
      let phong_light = light.colour * phong_scale as f32;
       // could also be extracted
      let phong_part  = phong_light * material.phong;
      result_colour = result_colour + phong_part;
    }
  }

  result_colour
}

fn trace_ray(scene: &scene::Scene, u: &Vec3, v: &Vec3, depth: uint) -> Colour {
  // Stop mirror ray recursion at a fixed depth
  if depth == 0 { return colour::BLACK; }

  // Find all objects that lie in the path of the ray for non -ve t
  let hits = scene.spheres.iter()
    .filter_map(|s| intersect_sphere(s, u, v))
    .collect::<Vec<HitS>>();
  
  // Trace the background colour in the abscence of any objects
  if hits.len() == 0 { return scene.background; }
  
  
  // Take the closest hit and extract the result
  let max_h = (&sphere!(), std::f64::MAX_VALUE, vector!(), false);
  let hit = hits.iter().fold(max_h, |s, h| { if h.val1() < s.val1() { *h } else { s }});

  let sphere = &hit.val0();
  let material = if hit.val3() { sphere.outer } else { sphere.inner };
  
  // Find the hit position, surface normal at the hit position, and reverse viewpoint vectors
  let hit_u = vec3::parametric_position(u, v, hit.val1() * 0.9999);
  let n_hat = hit.val2().as_vector().transform(&mat4::transpose(&sphere.inverse_t)).normalise();
  let v_hat = (*u - hit_u).normalise();

  // Do illumination using the Phong model
  let phong_colour = illuminate_hit(scene, &material, &hit_u, &n_hat, &v_hat);
  
  // Trace any reflections
  let mirror_colour = if colour::BLACK != material.mirror {
    let reflected_v = (n_hat * (2.0 * v_hat.dot(&n_hat)) - v_hat).normalise();
    let reflected_c = trace_ray(scene, &hit_u, &reflected_v, depth - 1);

    material.mirror * reflected_c
  } else {
    colour::BLACK
  };
  let result_colour = phong_colour + mirror_colour;


  // TODO transparency

  return result_colour.clamp();
}

/* Trace a position on screen given our progress so far, then update the colour array */
fn render_step(scene: &scene::Scene, colour_data: &mut [GLfloat], progress: uint) -> uint {
  // Some values from this function could be hoisted but the gains from doing so are probably
  // trivial compared to the cost of tracing

  // We fire 1 ray per pixel
  // Scene space is a right handed system, but with 0,0 in the centre so need to 
  // translate by w/2 and h/2 and then scale back to plane of -1,-1 to 1,1 
  // to transform from my screen space (RH system, origin in bottom left, w * h size

  const SSAA_SAMPLES: uint = 1;
  assert!(SSAA_SAMPLES > 0 && SSAA_SAMPLES % 2 == 1); // must be a +ve odd integer
  const SSAA_SAMPLES_SQ: f32 = SSAA_SAMPLES as f32 * SSAA_SAMPLES as f32;
  
  const DEPTH_MAX: uint = 5;
  assert!(DEPTH_MAX > 0); // 0 would produce an entirely black image

  let wx = scene.image_size.val0() as int;
  let wy = scene.image_size.val1() as int;

  let row = progress as int / (COLOUR_WIDTH as int * wy);
  let col = (progress as int % (COLOUR_WIDTH as int * wy)) / COLOUR_WIDTH as int;

  let hx = wx / 2;
  let hy = wy / 2;

  // ray start - this is our viewpoint
  let u = point!(0.0 0.0 1.0);
  
  
  // find the amount the direction changes per pixel
  // use the y-axis to fix the FOV so that 800x600 looks like 600x600 but showing more of the scene
  // instead of stretching the contents
  // let our viewplane in scene space be -1*wx/wy,-1,1 to 1*wx/wy,1,1
  let step_y = 1.0 / hy as f64;
  let step_half = step_y / 2.0;
  let sample_step = step_y / (SSAA_SAMPLES + 1) as f64;
  
  
  // ray direction
  // we add half the step so that the ray is in the centre of the sample rectangle
  // unless we are supersampling, in which case we want to sample a centred grid
  // starting from the bottom left
  let dx = (row as f64 - hx as f64) * step_y + 
    if SSAA_SAMPLES == 1 { step_half } else { sample_step };
  let dy = (col as f64 - hy as f64) * step_y +
    if SSAA_SAMPLES == 1 { step_half } else { sample_step };
  let dz = -1.0;
  
  // Uniform grid supersampling because I don't like random supersampling
  // TODO Try Poisson disc sampling
  let mut colour = colour::BLACK;
  for x in range(0, SSAA_SAMPLES) {
    for y in range(0, SSAA_SAMPLES) {
      let v = vector!(dx + (x as f64 * sample_step) dy + (y as f64 * sample_step) dz);
      colour = colour + trace_ray(scene, &u, &v, DEPTH_MAX);
    }
  }
  colour = colour * (1.0 / SSAA_SAMPLES_SQ);

  // Update the colour array with the result
  colour_data[progress] = colour.red;
  colour_data[progress + 1] = colour.green;
  colour_data[progress + 2] = colour.blue;

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

#[allow(dead_code)] // to silence test warnings
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
