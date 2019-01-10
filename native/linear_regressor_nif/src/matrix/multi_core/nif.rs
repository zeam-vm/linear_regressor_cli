#![allow(dead_code)]

use rustler::{Env, Term, NifResult, Encoder};
use matrix::multi_core as mc;

pub fn dot_product<'a>(env: Env<'a>, args: &[Term<'a>])-> NifResult<Term<'a>> {
  // Initialize Arguments
  // Decode to Vector
  let x: Vec<f64> = args[0].decode()?;
  let y: Vec<f64> = args[1].decode()?;

  // Return
  Ok(mc::dot_product(&x, &y).encode(env))
}

pub fn sub<'a>(env: Env<'a>, args: &[Term<'a>]) -> NifResult<Term<'a>> {
  let x: Vec<f64> = args[0].decode()?;
  let y: Vec<f64> = args[1].decode()?;
  
  Ok(mc::sub(&x, &y).encode(env))
}

pub fn emult<'a>(env: Env<'a>, args: &[Term<'a>]) -> NifResult<Term<'a>> {
  let x: Vec<f64> = args[0].decode()?;
  let y: Vec<f64> = args[1].decode()?;
  
  Ok(mc::emult(&x, &y).encode(env))
}

// pub fn new<'a>(env: Env<'a>, args: &[Term<'a>]) -> NifResult<Term<'a>> {
//   let first: i64 = args[0].decode()?;
//   let end: i64 = args[1].decode()?;

//   Ok(mc::new(first, end).encode(env))
// }

// pub fn zeros<'a>(env: Env<'a>, args: &[Term<'a>]) -> NifResult<Term<'a>> {
//   let len: usize = (args[0]).decode()?;
//   let zero_vec = vec![0; len];
//   Ok(zero_vec.encode(env))
// }