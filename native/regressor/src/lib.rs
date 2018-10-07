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

// use ndarray::arr2;

// type NifResult<T> = Result<T, Error>;
type Num = f64;

mod atoms {
  rustler_atoms! {
    atom ok;
    // atom error;
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
    ("_dot_product", 2, nif_dot_product),
    ("_zeros", 1, zeros),
    ("_new", 2, nif_new),
    ("_sub", 2, nif_sub),
    ("_emult", 2, nif_emult),
    ("_fit", 5, nif_fit), 
    ("_test", 2, test)
  ],
  None
}

pub fn dot_product(x: Vec<Num>, y: Vec<Num>) -> Num {
  // Main Process 
  let tuple = x.iter().zip(y.iter());
  // let a = tuple.iter();
  tuple.map(|t| t.0 * t.1).fold(0.0, |sum, i| sum + i)
}

pub fn sub(x: Vec<Num>, y: Vec<Num>) -> Vec<Num> {
  x.iter().zip(y.iter())
  .map(|t| t.0 - t.1)
  .collect()
}

pub fn sub2d(x: Vec<Vec<Num>>, y: Vec<Vec<Num>>) -> Vec<Vec<Num>>{
  x.iter().zip(y.iter())
  .map(|t| sub(t.0.to_vec(), t.1.to_vec()))
  .collect()
}

pub fn emult(x: Vec<Num>, y: Vec<Num>) -> Vec<Num> {
  x.iter().zip(y.iter())
  .map(|t| t.0 * t.1)
  .collect()
}

pub fn emult2d(x: Vec<Vec<Num>>, y: Vec<Vec<Num>>) -> Vec<Vec<Num>>{
  x.iter().zip(y.iter())
  .map(|t| emult(t.0.to_vec(), t.1.to_vec()))
  .collect()
}

// pub struct Matrix {
//   data: Vec<Num>, 
//   row_size: usize,
//   col_size: usize,
// }

// impl Matrix{
//   pub fn new(x: Vec<Vec<Num>>) -> Matrix{
//     let row = x.len();
//     let col = x[0].len();

//     let mut out: Vec<Num> = Vec::with_capacity(row*col);

//     // x.iter().map(|r| r.iter().map(|c| out.push(*c)));

//     for i in x{
//       for j in i{
//         out.push(j);
//       }
//     }

//     Matrix{
//       data: out,
//       row_size: row,
//       col_size: col,
//       transpose: false,
//     }
//   }

//   pub fn with_capacity(x: Vec<Vec<Num>>) -> Matrix{
//     let row = x.len();
//     let col = x[0].len();

//     Matrix{
//       data:Vec::with_capacity(row*col),
//       row_size: row,
//       col_size: col,
//       transpose: false,
//     }
//   }

//   pub fn data(&self) -> Vec<Num> {
//     self.data.clone()
//   }

//   pub fn split_first(&self) -> (&[Num], &[Num]){
//     self.data.as_slice().split_at(self.col_size)
//   }
// }

fn new_vec2(row: usize, col: usize, init: Num) -> Vec<Vec<Num>> {
  let mut col_vec: Vec<Num> = Vec::with_capacity(col);
  let mut ans: Vec<Vec<Num>> = Vec::with_capacity(row);

  col_vec.resize(col, init);
  ans.resize(row, col_vec);
  ans
}

pub fn transpose(x: Vec<Vec<Num>>) -> Vec<Vec<Num>> {
  swap_rows_cols(x)
} 

fn swap_rows_cols(x: Vec<Vec<Num>>) -> Vec<Vec<Num>> {
  let mut first :Vec<Num> = Vec::new();
  let mut rest :Vec<Vec<Num>> = Vec::new();
  let mut ans :Vec<Vec<Num>> = Vec::new();

  x.iter()
  .for_each(|r| {
    match r.as_slice().split_first() {
      Some((head, tail)) => {
        first.push(*head);
        rest.push(tail.to_vec());
      },
      None => {},
    }
  });
  
  ans.push(first);
  match rest.is_empty() {
    true => Vec::new(),
    false => {
      ans.append(&mut swap_rows_cols(rest));
      ans
    }
  }
}

// pub fn mult2d(x: Vec<Vec<Num>>, y: Vec<Vec<Num>>) -> Matrix {
//   let ans_row = x[0].len();
//   let ans_col = y[0].len();

//   let mut ans: Vec<Num> = Vec::with_capacity(ans_row*ans_col);
//   let mut sum = 0.0;

//   // println!("ans_row:{:?}", ans_row);
//   // println!("ans_col:{:?}", ans_col);

//   for i in 0..x.len(){
//     for k in 0..ans_col {
//       for j in 0..ans_row{    
//         sum = sum + x[i][j]*y[j][k];
//       }            
//       println!("{:?}", sum);
//       ans.push(sum);
//       sum = 0.0;
//     }        
//   }

//   Matrix{
//     data: ans,
//     row_size: ans_row,
//     col_size: ans_col,
//     transpose: false,
//   }
// }

pub fn mult (x: Vec<Vec<Num>>, y: Vec<Vec<Num>>) -> Vec<Vec<Num>> {
  let ty = transpose(y);

  x.iter()
  .map(|i| {
    ty.iter()
    .map(|j| dot_product(i.to_vec(), j.to_vec()))
    .collect()
  })
  .collect()
}

fn test<'a>(env: Env<'a>, args: &[Term<'a>])-> NifResult<Term<'a>> {
  let x: Vec<Vec<Num>> = args[0].decode()?;
  let y: Vec<Vec<Num>> = args[1].decode()?;

  Ok(atoms::ok().encode(env))
}

fn nif_fit<'a>(env: Env<'a>, args: &[Term<'a>])-> NifResult<Term<'a>> {
  let x: Vec<Vec<Num>> = args[0].decode()?;
  let y: Vec<Vec<Num>> = args[1].decode()?;
  let tx = transpose(x.clone());
  let theta: Vec<Vec<Num>> = args[2].decode()?;
  let alpha: Num = try!(args[3].decode());
  let iterations: i64 = try!(args[4].decode());

  let m = y.len() as Num;
  let trans_theta = transpose(theta.clone());
  let (left, right) = (theta.len(), theta[0].len());
  let a = new_vec2(left, right, alpha / m);

  // println!("m:{:?}", m);
  // println!("tx:{:?}", tx.iter());
  // println!("{:?}, {:?}", left, right);
  // println!("{:?}", a);

  // println!("theta{:?}", theta.iter());
  // println!("trans_theta:{:?}", trans_theta);

  let ans = (0..iterations)
  .fold( theta, |theta, _iteration|{
    let x = x.clone();
    let y = y.clone();
    let tx = tx.clone();
    let a = a.clone();
    let trans_theta = transpose(theta.clone());

    let mut d = mult( tx, sub2d( mult(x, theta.clone()), y ) );
    d = emult2d(d, a);
    sub2d(theta, d)
  });

  Ok(ans.encode(env))
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

fn nif_emult<'a>(env: Env<'a>, args: &[Term<'a>]) -> NifResult<Term<'a>> {
  let x: Vec<Num> = args[0].decode()?;
  let y: Vec<Num> = args[1].decode()?;
  
  Ok(emult(x, y).encode(env))
}

fn new(first: i64, end: i64) -> Vec<i64> {
  (first..end).collect()
}

fn nif_new<'a>(env: Env<'a>, args: &[Term<'a>]) -> NifResult<Term<'a>> {
  let first: i64 = try!(args[0].decode());
  let end: i64 = try!(args[1].decode());

  Ok(new(first, end).encode(env))
}

fn zeros<'a>(env: Env<'a>, args: &[Term<'a>]) -> NifResult<Term<'a>> {
  let len: usize = (args[0]).decode()?;
  let zero_vec = vec![0; len];
  Ok(zero_vec.encode(env))
}