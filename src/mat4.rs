use std::num::FloatMath;
use std::f64 as stdf64;

use vec4::{Vec4,normalise};

pub const DIM: uint = 4;

pub type Matrix = [[f64, ..DIM], ..DIM];
// if I want to add traits for this I need to use a newtype... and then manually add all the inner type traits
//... blergh

pub fn identity() -> Matrix {
  [[1.0, 0.0, 0.0, 0.0],
   [0.0, 1.0, 0.0, 0.0],
   [0.0, 0.0, 1.0, 0.0],
   [0.0, 0.0, 0.0, 1.0]]
}

fn zero() -> Matrix {
  [[0.0, ..DIM], ..DIM]
}

pub fn matrix_print(a: &Matrix, title: &str) {
    println!("{}", title);

    // there has got to be a way to do this purely using iter() and map() and co.
    for row in a.iter() {
        let str: Vec<String> = row.iter().map(|&v| stdf64::to_str_exact(v, 3u)).collect();
        println!("{}", str);
    }
    println!("");
}

pub fn multiply(a: &Matrix, b: &Matrix) -> Matrix {
    let mut result = zero();
    for row in range(0, DIM) {
        for col in range(0, DIM) {
            for k in range(0, DIM) {
                result[row][col] += a[row][k] * b[k][col];
            }
        }
    }
    result
}

/*pub fn add(a: &Matrix, b: &Matrix) -> Matrix {
    let mut result = zero();
    for row in range(0, DIM) {
        for col in range(0, DIM) {
            result[row][col] = a[row][col] + b[row][col];
        }
    }
    result
}
*/
pub fn transpose(a: &Matrix) -> Matrix {
    let mut result = zero();
    for row in range(0, DIM) {
        for col in range(0, DIM) {
            result[row][col] = a[col][row];
        }
    }
    result
}

pub fn translate(v: &Vec4) -> Matrix {
    let mut result = identity();
    result[0][3] = v.x;
    result[1][3] = v.y;
    result[2][3] = v.z;

    result
}

pub fn scale(v: &Vec4) -> Matrix {
    let mut result = identity();
    result[0][0] = v.x;
    result[1][1] = v.y;
    result[2][2] = v.z;

    result
}

pub fn rotate(axis: &Vec4, rads: f64) -> Matrix {
    let mut result = identity();

    let sinr = FloatMath::sin(rads);
    let cosr = FloatMath::cos(rads);
    let u = normalise(axis);

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
