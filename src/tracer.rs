use std::{rand,f64};
use std::num::{Float,FloatMath};
use mat4;
use vec3;
use vec3::{Vec3,DotProduct,Normalise,Transform,AsVector,Apply};
// TODO clean up trait usage once UFCS has been implemented in Rust
use colour;
use colour::{Colour,Clamp};
use scene;
use scene::{Sphere,Material};


// Core ray tracing routines


// Types
type TestOpt = Option<(f64, f64)>;
type HitS<'a> = (&'a scene::Sphere, f64, Vec3, bool); // REALLY should be a struct at this point
type HitSOpt<'a> = Option<HitS<'a>>;

// Private functions
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

  let p = (bsq - ac4).sqrt();
  
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

// Calculate what factor to colour a position by, using random sampling to soften shadows 
fn soft_shadow_scale(scene: &scene::Scene, light: &scene::Light, hit_u: &Vec3) -> f64 {
  const SOFT_SHADOW_SAMPLES: uint = 20;
  const SOFT_SHADOW_COEFF: f64 = 0.05;
  assert!(SOFT_SHADOW_SAMPLES > 0);
  
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

// Illuminate a hit using the Phong illumination model
// hit_u is the hit position
// n_hat is the (unit) surface normal at the hit position
// v_hat is the (unit) vector from the hit position back to the viewpoint
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
      let phong_scale = rv.powi(material.phong_n as i32);
      let phong_light = light.colour * phong_scale as f32;
      // could also be extracted
      let phong_part  = phong_light * material.phong;
      result_colour = result_colour + phong_part;
    }
  }

  result_colour
}

// Public functions
pub fn trace_ray(scene: &scene::Scene, u: &Vec3, v: &Vec3, depth: uint) -> Colour {
  // Stop mirror ray recursion at a fixed depth
  if depth == 0 { return colour::BLACK; }

  // Find all objects that lie in the path of the ray for non -ve t
  let hits = scene.spheres.iter()
    .filter_map(|s| intersect_sphere(s, u, v))
    .collect::<Vec<HitS>>();
  
  // Trace the background colour in the abscence of any objects
  if hits.len() == 0 { return scene.background; }
  
  
  // Take the closest hit and extract the result
  let max_h = (&sphere!(), f64::MAX_VALUE, vector!(), false);
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

// Unit tests
#[cfg(test)]
mod test {
  use vec3::Vec3;
  
  #[test]
  fn trace_intersect_unit() {
    let u_centre = point!(0.0 0.0 1.0);
    let m1 = -1.0;
    let v = vector!(0.0 0.0 m1);
    let u_bottom_left = vector!(m1 m1 m1);
    let u_top_right = vector!(1.0 1.0 m1);

    assert!(None != super::intersect_fixed_sphere(&u_centre, &v, 1.0));
    assert!(None == super::intersect_fixed_sphere(&u_bottom_left, &v, 1.0));
    assert!(None == super::intersect_fixed_sphere(&u_top_right, &v, 1.0));
  }
}
