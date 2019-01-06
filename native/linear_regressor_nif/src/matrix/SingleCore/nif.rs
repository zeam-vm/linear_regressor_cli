
use rustler::{Env, Term, NifResult, Encoder};

use Matrix::SingleCore as sc;

pub fn dot_product<'a>(env: Env<'a>, args: &[Term<'a>])-> NifResult<Term<'a>> {
  // Initialize Arguments
  // Decode to Vector
  let x: Vec<f64> = args[0].decode()?;
  let y: Vec<f64> = args[1].decode()?;

  // Return
  Ok(sc::dot_product(&x, &y).encode(env))
}

pub fn sub<'a>(env: Env<'a>, args: &[Term<'a>]) -> NifResult<Term<'a>> {
  let x: Vec<f64> = args[0].decode()?;
  let y: Vec<f64> = args[1].decode()?;
  
  Ok(sc::sub(&x, &y).encode(env))
}

pub fn emult<'a>(env: Env<'a>, args: &[Term<'a>]) -> NifResult<Term<'a>> {
  let x: Vec<f64> = args[0].decode()?;
  let y: Vec<f64> = args[1].decode()?;
  
  Ok(sc::emult(&x, &y).encode(env))
}

pub fn new<'a>(env: Env<'a>, args: &[Term<'a>]) -> NifResult<Term<'a>> {
  let first: i64 = args[0].decode()?;
  let end: i64 = args[1].decode()?;

  Ok(sc::new(first, end).encode(env))
}

pub fn zeros<'a>(env: Env<'a>, args: &[Term<'a>]) -> NifResult<Term<'a>> {
  let len: usize = (args[0]).decode()?;
  let zero_vec = vec![0; len];
  Ok(zero_vec.encode(env))
}