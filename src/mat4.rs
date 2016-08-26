use vec3::{Vec3,Normalise};


// 4x4 Matrix module

// Constants

pub const DIM: usize = 4;

// Types

// If I want to add traits for this I need to use a newtype...
// and then manually add all the inner type traits...
// TODO wait for Rust to fix this
pub type Matrix = [[f64; DIM]; DIM];

// Private functions
fn zero() -> Matrix {
  [[0.0; DIM]; DIM]
}

// Public functions
pub fn identity() -> Matrix {
  [[1.0, 0.0, 0.0, 0.0],
   [0.0, 1.0, 0.0, 0.0],
   [0.0, 0.0, 1.0, 0.0],
   [0.0, 0.0, 0.0, 1.0]]
}

// Can't implement Show trait for this type yet -- see type declaration
pub fn matrix_print(a: &Matrix, title: &str) {
  println!("{}", title);

  for row in a.iter() { // what happened to stdf64::to_str_exact(v, 3) ???
    let str: Vec<String> = row.iter().map(|&v| v.to_string()).collect();
    println!("{:?}", str);
  }
  println!("");
}

pub fn multiply(a: &Matrix, b: &Matrix) -> Matrix {
  let mut result = zero();
  for row in 0.. DIM {
    for col in 0.. DIM {
      for k in 0.. DIM {
        result[row][col] += a[row][k] * b[k][col];
      }
    }
  }
  result
}

pub fn transpose(a: &Matrix) -> Matrix {
  let mut result = zero();
  for row in 0.. DIM {
    for col in 0.. DIM {
      result[row][col] = a[col][row];
    }
  }
  result
}

pub fn new_translation(v: &Vec3) -> Matrix {
  let mut result = identity();
  result[0][3] = v.x;
  result[1][3] = v.y;
  result[2][3] = v.z;

  result
}

pub fn new_scale(v: &Vec3) -> Matrix {
  let mut result = identity();
  result[0][0] = v.x;
  result[1][1] = v.y;
  result[2][2] = v.z;

  result
}

pub fn new_rotation(axis: &Vec3, rads: f64) -> Matrix {
  let mut result = identity();

  let sinr = rads.sin();
  let cosr = rads.cos();
  let u = axis.normalise();

  // TODO sq macro if there isn't already one
  let xx = u.x * u.x;
  let yy = u.y * u.y;
  let zz = u.z * u.z;

  let xy = u.x * u.y;
  let yz = u.y * u.z;
  let xz = u.x * u.z;
  let lcosr = 1.0 - cosr;

  // TODO could do even less multiplications
  result[0][0] = xx * lcosr + cosr;
  result[0][1] = xy * lcosr - u.z * sinr;
  result[0][2] = xz * lcosr + u.y * sinr;

  result[1][0] = xy * lcosr + u.z * sinr;
  result[1][1] = yy * lcosr + cosr;
  result[1][2] = yz * lcosr - u.x * sinr;

  result[2][0] = xz * lcosr - u.y * sinr;
  result[2][1] = yz * lcosr + u.x * sinr;
  result[2][2] = zz * lcosr + cosr;

  result
}

/* Unit tests */
#[cfg(test)]
mod test {
  use super::*;
  use std::f64::consts;
  use vec3::Vec3;

  #[test]
  fn mat4_zero() {
    let m = super::zero();
    let s = m.iter().flat_map(|x| x.iter()).fold(0.0, |sum, &e| sum + e);
    assert!(s == 0.0);
  }

  #[test]
  fn mat4_identity() {
    let m = super::identity();
    let s = m.iter().flat_map(|x| x.iter()).fold(0.0, |sum, &e| sum + e);

    for x in 0.. DIM {
      assert!(m[x][x] == 1.0);
    }

    assert!(s == DIM as f64);
  }

  #[test]
  fn mat4_transpose() {
    let mut m = super::identity();
    m[DIM - 1][0] = 5.0;
    m[0][DIM - 1] = 4.0;

    let n = super::transpose(&m);
    let s = n.iter().flat_map(|x| x.iter()).fold(0.0, |sum, &e| sum + e);


    for x in 0.. DIM {
      assert!(n[x][x] == 1.0);
    }
    assert!(n[DIM - 1][0] == 4.0);
    assert!(n[0][DIM - 1] == 5.0);
    assert!(s == DIM as f64 + 9.0);
  }

  #[test]
  fn mat4_rotate() {
    let v = vector!(1.0, 0.0, 0.0);
    let r = consts::PI / 2.0;
    let m = super::new_rotation(&v, r);

    let s = m.iter().flat_map(|x| x.iter()).fold(0.0, |sum, &e| sum + e);

    assert!(m[0][0] == 1.0);
    assert!(m[1][2] == -1.0);
    assert!(m[2][1] == 1.0);
    assert!(m[3][3] == 1.0);
    assert!(s == 2.0);
  }

  #[test]
  fn mat4_translate() {
    let v = vector!(1.0, 2.0, 3.0);
    let m = super::new_translation(&v);

    let s = m.iter().flat_map(|x| x.iter()).fold(0.0, |sum, &e| sum + e);

    for x in 0.. DIM - 1 {
      assert!(m[x][DIM - 1] == v[x]);
    }
    assert!(m[DIM - 1][DIM - 1] == 1.0);
    assert!(s == 10.0);
  }

  #[test]
  fn mat4_scale() {
    let v = vector!(1.0, 2.0, 3.0);
    let m = super::new_scale(&v);

    let s = m.iter().flat_map(|x| x.iter()).fold(0.0, |sum, &e| sum + e);

    for x in 0.. DIM - 1 {
      assert!(m[x][x] == v[x]);
    }
    assert!(m[DIM - 1][DIM - 1] == 1.0);
    assert!(s == 7.0)
  }

  #[test]
  fn mat4_multiply() {
    let m = super::identity();
    let n = [[1.0, 2.0, 3.0, 4.0],
             [5.0, 6.0, 7.0, 8.0],
             [8.0, 7.0, 6.0, 5.0],
             [4.0, 3.0, 2.0, 1.0]];

    let o = super::multiply(&n, &m);

    for r in 0.. DIM {
      for c in 0.. DIM {
        assert!(o[r][c] == n[r][c]);
      }
    }
  }
}
