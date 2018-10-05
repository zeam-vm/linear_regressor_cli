#[macro_use] extern crate rustler;
// #[macro_use] extern crate rustler_codegen;
#[macro_use] extern crate lazy_static;

use rustler::{Env, Term, NifResult, Encoder};
// use rustler::env::{OwnedEnv, SavedTerm};
// use rustler::types::atom::Atom;
// use rustler::types::list::ListIterator;
// use rustler::types::map::MapIterator;

// use rustler::types::tuple::make_tuple;
// use std::ops::Range;
// use std::ops::RangeInclusive;

// use rayon::prelude::*;

// type NifResult<T> = Result<T, Error>;
type Num = i64;

mod atoms {
    rustler_atoms! {
        atom ok;
        atom error;
        //atom __true__ = "true";
        //atom __false__ = "false";
    }
}

// macro_rules! attach_nif {
//     ( $i:item, $x:expr ) => ("item", x, item)
// }

rustler_export_nifs! {
    "Elixir.NifRegressor",
    [
        //("Elixir's func, number of arguments, Rust's func)
        ("dot_product", 2, nif_dot_product),
        ("zeros", 1, zeros),
        ("new", 2, new),
        ("sub", 2, nif_sub),
        ("emult", 2, nif_emult),
        ("fit", 5, nif_fit), 
        ("test", 2, test)
    ],
    None
}

pub fn dot_product(x: Vec<Num>, y: Vec<Num>) -> Num {
    // Main Process 
    let tuple = x.iter().zip(y.iter());
    tuple.map(|t| t.0 * t.1).fold(0, |sum, i| sum +i)
}

pub fn sub(x: Vec<Num>, y: Vec<Num>) -> Vec<Num> {
    x.iter().zip(y.iter())
    .map(|t| t.0 - t.1)
    .collect()
}

pub fn emult(x: Vec<Num>, y: Vec<Num>) -> Vec<Num> {
    x.iter().zip(y.iter())
    .map(|t| t.0 * t.1)
    .collect()

}

pub struct Matrix {
    data: Vec<Num>, 
    row_size: usize,
    col_size: usize,
    transpose: bool,
}

impl Matrix{
  pub fn data(&self) -> Vec<Num> {
    self.data.clone()
  }
}

// pub fn new_matrix(x: Vec<Vec<Num>>, init: Num) -> Matrix{
//     let row = x.len();
//     let col = x[0].len();    
// }

pub fn transpose(x: Vec<Vec<Num>>) -> Matrix {

}

pub fn mult2d(x: Vec<Vec<Num>>, y: Vec<Vec<Num>>) -> Matrix {
  let ans_row = x[0].len();
  let ans_col = y[0].len();

  let mut ans: Vec<Num> = Vec::with_capacity(ans_row*ans_col);
  let mut sum = 0;

  println!("ans_row:{:?}", ans_row);
  println!("ans_col:{:?}", ans_col);

  for i in 0..x.len(){
    for k in 0..ans_col {
      for j in 0..ans_row{    
        sum = sum + x[i][j]*y[j][k];
      }            // println!("{:?}", sum);
      ans.push(sum);
      sum = 0;
    }        
  }

  Matrix{
      data: ans,
      row_size: ans_row,
      col_size: ans_col,
      transpose: false,
  }
}

// pub fn transpose (x: Vec<Vec<i64>>) -> Vec<Vec<i64>> {
//     let x

//     let mut ans = x.clone();

//     let tuple = x.iter()
//     ans.reverse();
//     ans
// }

fn test<'a>(env: Env<'a>, args: &[Term<'a>])-> NifResult<Term<'a>> {
    let mut x: Vec<Vec<Num>> = args[0].decode()?;
    let y: Vec<Vec<Num>> = args[1].decode()?;

    // if x[0].len() != y.len(){
    //     Err(atoms::error().encode(env));
    // }

    for i in 0..x.len(){
        for j in 0..x[0].len(){
            println!("{:?}", &x[i][j]);
        }
    }



    let mat: Matrix = mult2d(x, y);

    // for i in mat.data.clone().iter(){  
    //     println!("{:?}", i);
    // }

    Ok(atoms::ok().encode(env))
}

fn nif_fit<'a>(env: Env<'a>, args: &[Term<'a>])-> NifResult<Term<'a>> {
    let x: Vec<Vec<Num>> = args[0].decode()?;
    println!("{:?}", x);
    
    Ok(atoms::ok().encode(env))
}

fn nif_dot_product<'a>(env: Env<'a>, args: &[Term<'a>])-> NifResult<Term<'a>> {
    // Initialize Arguments
    // Decode to Vector
    let x: Vec<Num> = args[0].decode()?;
    let y: Vec<Num> = args[1].decode()?;

    // Return
    Ok(dot_product(x, y).encode(env))
}

fn nif_sub<'a>(env: Env<'a>, args: &[Term<'a>]) -> NifResult<Term<'a>> {
    let x: Vec<Num> = args[0].decode()?;
    let y: Vec<Num> = args[1].decode()?;
    
    Ok(sub(x, y).encode(env))
}

// Multiplication for each elements of vector
fn nif_emult<'a>(env: Env<'a>, args: &[Term<'a>]) -> NifResult<Term<'a>> {
    let x: Vec<Num> = args[0].decode()?;
    let y: Vec<Num> = args[1].decode()?;
    
    Ok(emult(x, y).encode(env))
}

fn new<'a>(env: Env<'a>, args: &[Term<'a>]) -> NifResult<Term<'a>> {
    let first: i64 = try!(args[0].decode());
    let end: i64 = try!(args[1].decode());
    
    let vec : Vec<i64> = (first..end).collect();

    Ok(vec.encode(env))
}

fn zeros<'a>(env: Env<'a>, args: &[Term<'a>]) -> NifResult<Term<'a>> {
    let len: usize = (args[0]).decode()?;
    let zero_vec = vec![0; len];
    Ok(zero_vec.encode(env))
}

// fn fit<'a>(env: Env<'a>, args: &[Term<'a>]) -> NifResult<Term<'a>> {
//     // let x Vec=  
// }
