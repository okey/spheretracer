use gl;
use gl::types::*;

use std::ptr;
use std::str;

// Vertex shader source
pub static VS_SRC: &'static str =
"#version 150\n\
in vec2 position;\n\
in vec4 vertex_colour;\n\
out vec4 fragment_colour;\n\
uniform mat4 mvp;\n\
void main() {\n\
gl_Position = mvp * vec4(position, 0.0, 1.0);\n\
fragment_colour = vertex_colour;\n\
}";

// Fragment shader source
pub static FS_SRC: &'static str =
"#version 150\n\
in vec4 fragment_colour;\n\
out vec4 out_colour;\n\
void main() {\n\
out_colour = fragment_colour;\n\
}";


// Compile a shader and panic! if something goes wrong
pub fn compile_shader(src: &str, ty: GLenum) -> GLuint {
  let shader;
  unsafe {
    shader = gl::CreateShader(ty);

    src.with_c_str(|ptr| gl::ShaderSource(shader, 1, &ptr, ptr::null()));
    gl::CompileShader(shader);

    let mut status = gl::FALSE as GLint;
    gl::GetShaderiv(shader, gl::COMPILE_STATUS, &mut status);

    if status != (gl::TRUE as GLint) {
      let mut len = 0;
      gl::GetShaderiv(shader, gl::INFO_LOG_LENGTH, &mut len);
      let mut buf = Vec::from_elem(len as uint - 1, 0u8); // -1 for null terminator
      gl::GetShaderInfoLog(shader, len, ptr::null_mut(), buf.as_mut_ptr() as *mut GLchar);
      panic!("{}", str::from_utf8(buf.as_slice()).expect("ShaderInfoLog not valid utf8"));
    }
  }
  shader
}

// Link the program and panic! if something goes wrong
pub fn link_program(vs: GLuint, fs: GLuint) -> GLuint {
  unsafe {
    let program = gl::CreateProgram();
    gl::AttachShader(program, vs);
    gl::AttachShader(program, fs);
    gl::LinkProgram(program);

    let mut status = gl::FALSE as GLint;
    gl::GetProgramiv(program, gl::LINK_STATUS, &mut status);

    if status != (gl::TRUE as GLint) {
      let mut len: GLint = 0;
      gl::GetProgramiv(program, gl::INFO_LOG_LENGTH, &mut len);
      let mut buf = Vec::from_elem(len as uint - 1, 0u8); // -1 for null terminator
      gl::GetProgramInfoLog(program, len, ptr::null_mut(), buf.as_mut_ptr() as *mut GLchar);
      panic!("{}", str::from_utf8(buf.as_slice()).expect("ProgramInfoLog not valid utf8"));
    }
    program
  }
}
