#![feature(slicing_syntax)]
#![feature(macro_rules)]
#![feature(phase)]
#![feature(globs)]
#![feature(if_let)]

#[phase(plugin)]
extern crate regex_macros;

extern crate regex;
extern crate core;
extern crate gl;
extern crate glfw;

use std::{os,mem,ptr};
use std::path::Path;

use gl::types::*;
use glfw::{Context,OpenGlProfileHint,WindowHint,WindowMode};

use vec3::{Vec3,Transform}; // TODO clean up trait usage once UFCS has been implemented in Rust

// NOTE module definition order matters for macro exports!
mod colour;
mod vec3;
mod mat4;
mod scene;
mod sceneio;
mod tracer;

mod shaders;


// Constants

// The number of components expected per value (colour or position) in the data arrays drawn
const COLOUR_WIDTH:uint = 4;
const VERTEX_WIDTH:uint = 2;


// Functions

// Make an orthographic matrix to transform from NDC to world (scene) space
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
  glfw.window_hint(WindowHint::ContextVersion(3, 2));
  glfw.window_hint(WindowHint::OpenglForwardCompat(true));
  glfw.window_hint(WindowHint::OpenglProfile(OpenGlProfileHint::Core));
  glfw.window_hint(WindowHint::Resizable(false));

  // TODO support optional fullscreen rendering
  let (window, _) = glfw.create_window(wx as u32, wy as u32, "OpenGL", WindowMode::Windowed)
    .expect("Failed to create GLFW window.");
  let (real_wx, real_wy) =
    match window.get_size() {
      (x, y) if x as uint != wx || y as uint != wy => {
        println!("Warning! Size {}x{} not windowable, rendering {}x{} instead",
                 wx, wy, x, y);
        (x as uint, y as uint)
      },
      
      _ => (wx, wy)
    };

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
  let wy2 = real_wy * VERTEX_WIDTH;
  let vertex_data_vec: Vec<GLfloat> = Vec::from_fn(real_wx * wy2, |n| {
    let x = n / wy2;
    let y = n % wy2;
    let v = if y % 2 == 0 { x as f32 } else { (y / VERTEX_WIDTH) as f32};
    v + 0.5
  });
  let vertex_data = vertex_data_vec.as_slice();
  
  // Add a colour for each vertex
  let mut cbo = 0;
  let wy4 = real_wy * COLOUR_WIDTH;
  let mut colour_data_vec: Vec<GLfloat> = Vec::from_fn(real_wx * wy4, |n| {
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
    let proj = make_gl_ortho_mat(real_wx, real_wy);
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
  let chunk_size = real_wy * 4;
  let max_chunk = colour_data.len() / chunk_size;
  
  let mut render_progress = 0;
  // TODO better GL performance for scenes that render quickly
  // Streaming, mapping the buffer, or writing to a dynamic texture are a few options
  while !window.should_close() {
    glfw.poll_events();
    
    while chunk <= max_chunk && render_progress < chunk_size * chunk {
      render_progress = render_step(scene, colour_data, render_progress, real_wx, real_wy);
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

// Trace a position on screen given our progress so far, then update the colour array
fn render_step(scene: &scene::Scene, colour_data: &mut [GLfloat], progress: uint,
               wx: uint, wy: uint) -> uint {
  // Some values from this function could be hoisted but the gains from doing so are probably
  // trivial compared to the cost of tracing

  // We fire 1 ray per pixel
  // Scene space is a right handed system, but with 0,0 in the centre so need to 
  // translate by w/2 and h/2 and then scale back to plane of -1,-1 to 1,1 
  // to transform from my screen space (RH system, origin in bottom left, w * h size

  const SSAA_SAMPLES: uint = 3;
  assert!(SSAA_SAMPLES > 0 && SSAA_SAMPLES % 2 == 1); // must be a +ve odd integer
  const SSAA_SAMPLES_SQ: f32 = SSAA_SAMPLES as f32 * SSAA_SAMPLES as f32;
  
  const DEPTH_MAX: uint = 5;
  assert!(DEPTH_MAX > 0); // 0 would produce an entirely black image

  let row =  progress / (COLOUR_WIDTH * wy);
  let col = (progress % (COLOUR_WIDTH * wy)) / COLOUR_WIDTH;

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
      colour = colour + tracer::trace_ray(scene, &u, &v, DEPTH_MAX);
    }
  }
  colour = colour * (1.0 / SSAA_SAMPLES_SQ);

  // Update the colour array with the result
  for idx in range(0, std::cmp::min(colour::CHANNELS, COLOUR_WIDTH)) {
    colour_data[progress + idx] = colour[idx];
  }

  progress + COLOUR_WIDTH
}

fn load_scene_or_fail(filename: &Path) -> scene::Scene {
  let scene = match sceneio::read_scene(filename) {
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

#[allow(dead_code)] // to silence unit test compile warnings
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
