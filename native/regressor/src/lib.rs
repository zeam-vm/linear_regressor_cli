#[macro_use] extern crate rustler;
// #[macro_use] extern crate rustler_codegen;
#[macro_use] extern crate lazy_static;

use rustler::{Env, Term, NifResult, Encoder};
use rustler::env::{OwnedEnv, SavedTerm};
// use rustler::types::atom::Atom;
// use rustler::types::list::ListIterator;
// use rustler::types::map::MapIterator;

use rustler::types::tuple::make_tuple;
// use std::ops::Range;
// use std::ops::RangeInclusive;

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
  ],
  None
}

pub fn dot_product(x: Vec<Num>, y: Vec<Num>) -> Num {
  x.iter().zip(y.iter())
  .map(|t| t.0 * t.1).fold(0.0, |sum, i| sum + i)
}

pub fn sub(x: Vec<Num>, y: Vec<Num>) -> Vec<Num> {
  x.iter().zip(y.iter())
  .map(|t| t.0 - t.1)
  .collect()
}

pub fn sub2d(x: &Vec<Vec<Num>>, y: &Vec<Vec<Num>>) -> Vec<Vec<Num>>{
  x.iter().zip(y.iter())
  .map(|t| sub(t.0.to_vec(), t.1.to_vec()))
  .collect()
}

pub fn emult(x: Vec<Num>, y: Vec<Num>) -> Vec<Num> {
  x.iter().zip(y.iter())
  .map(|t| t.0 * t.1)
  .collect()
}

pub fn emult2d(x: &Vec<Vec<Num>>, y: &Vec<Vec<Num>>) -> Vec<Vec<Num>>{
  x.iter().zip(y.iter())
  .map(|t| emult(t.0.to_vec(), t.1.to_vec()))
  .collect()
}

fn new_vec2(row: usize, col: usize, init: Num) -> Vec<Vec<Num>> {
  let mut col_vec: Vec<Num> = Vec::with_capacity(col);
  let mut ans: Vec<Vec<Num>> = Vec::with_capacity(row);

  col_vec.resize(col, init);
  ans.resize(row, col_vec);
  ans
}

pub fn transpose(x: &Vec<Vec<Num>>) -> Vec<Vec<Num>> {
  let row :usize = x.len();
  let col :usize = x[0].len();
  (0..col)
  .map(|c| {
    (0..row)
    .map( |r| x[r][c] )
    .collect()
  })
  .collect()
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

pub fn mult (x: &Vec<Vec<Num>>, y: &Vec<Vec<Num>>) -> Vec<Vec<Num>> {
  let ty = transpose(y);

  x.iter()
  .map(|i| {
    ty.iter()
    .map(|j| dot_product(i.to_vec(), j.to_vec()))
    .collect()
  })
  .collect()
}

fn nif_fit<'a>(env: Env<'a>, args: &[Term<'a>])-> NifResult<Term<'a>> {
  let pid = env.pid();
  let mut my_env = OwnedEnv::new();

  let saved_list = my_env.run(|env| -> NifResult<SavedTerm> {
    let _x = args[0].in_env(env);
    let _y = args[1].in_env(env);
    let theta = args[2].in_env(env);
    let alpha = args[3].in_env(env);
    let iterations = args[4].in_env(env);
    Ok(my_env.save(make_tuple(env, &[_x, _y, theta, alpha, iterations])))
  })?;

  std::thread::spawn(move ||  {
    my_env.send_and_clear(&pid, |env| {
      let result: NifResult<Term> = (|| {
        let tuple = saved_list
        .load(env).decode::<(
          Term, 
          Term,
          Term,
          Num,
          i64)>()?; 
        
        let x: Vec<Vec<Num>> = try!(tuple.0.decode());
        let y: Vec<Vec<Num>> = try!(tuple.1.decode());
        let theta: Vec<Vec<Num>> = try!(tuple.2.decode());
        let alpha: Num = tuple.3;
        let iterations: i64 = tuple.4;

        let tx = transpose(&x);
        let m = y.len() as Num;
        let (left, right) = (theta.len(), theta[0].len());
        let a = new_vec2(left, right, alpha / m);

        let ans = (0..iterations)
          .fold( theta, |theta, _iteration|{
            //let x = x.clone();
            //let y = y.clone();
            //let tx = tx.clone();
            //let a = a.clone();

           sub2d(&theta, &emult2d(&mult( &tx, &sub2d( &mult( &x, &theta ), &y ) ), &a))
          });

        Ok(ans.encode(env))
      })();
      match result {
          Err(_err) => env.error_tuple("test failed".encode(env)),
          Ok(term) => term
      }  
    });
  });
  Ok(atoms::ok().to_term(env))
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