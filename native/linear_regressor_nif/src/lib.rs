#[macro_use] extern crate rustler;
// #[macro_use] extern crate rustler_codegen;
#[macro_use] extern crate lazy_static;
extern crate rayon;
extern crate time;
extern crate num_cpus;

use rustler::{Env, Term, NifResult, Encoder};
use rustler::env::{OwnedEnv, SavedTerm};
use rustler::types::tuple::make_tuple;

use rayon::prelude::*;
use rayon::ThreadPool;
use rayon::ThreadPoolBuildError;

type Num = f64;

mod atoms {
  rustler_atoms! {
    atom ok;
    // atom error;
    //atom __true__ = "true";
    //atom __false__ = "false";
  }
}

rustler_export_nifs! {
  "Elixir.LinearRegressorNif",
  [
    //("Elixir's func, number of arguments, Rust's func)
    ("_dot_product", 2, nif_dot_product),
    ("_zeros", 1, zeros),
    ("_new", 2, nif_new),
    ("_sub", 2, nif_sub),
    ("_emult", 2, nif_emult),
    ("_fit", 4, nif_fit),
    ("_rayon_fit", 4, rayon_fit), 
    ("_nif_benchmark", 4, nif_benchmark),
  ],
  None
}

pub fn dot_product(x: &Vec<Num>, y: &Vec<Num>) -> Num {
  // x.iter().zip(y.iter())
  // .map(|t| t.0 * t.1)
  // .fold(0.0, |sum, i| sum + i)

  let row :usize = x.len();
  
  (0..row)
  .map(|r| x[r]*y[r] )
  .fold(0.0, |sum, i| sum + i)
}

pub fn sub(x: &Vec<Num>, y: &Vec<Num>) -> Vec<Num> {
  // x.iter().zip(y.iter())
  // .map(|t| t.0 - t.1)
  // .collect()
  let row :usize = x.len();
  
  (0..row)
  .map(|r| x[r]-y[r] )
  .collect()
}

pub fn sub2d(x: &Vec<Vec<Num>>, y: &Vec<Vec<Num>>) -> Vec<Vec<Num>>{
  // x.iter().zip(y.iter())
  // .map(|t| sub(&t.0 as &Vec<Num>, &t.1 as &Vec<Num>))
  // .collect()

  let row :usize = x.len();
  let col :usize = x[0].len();

  (0..row)
  .map(|r| {
    (0..col)
    .map( |c| x[r][c]-y[r][c])
    .collect()
  })
  .collect()
}

pub fn emult(x: &Vec<Num>, y: &Vec<Num>) -> Vec<Num> {
  // x.iter().zip(y.iter())
  // .map(|t| t.0 * t.1)
  // .collect()

  let row :usize = x.len();
  
  (0..row)
  .map(|r| x[r]*y[r] )
  .collect()
}

pub fn emult2d(x: &Vec<Vec<Num>>, y: &Vec<Vec<Num>>) -> Vec<Vec<Num>>{
  // x.iter().zip(y.iter())
  // .map(|t| emult(&t.0 as &Vec<Num>, &t.1 as &Vec<Num>))
  // .collect()

  let row :usize = x.len();
  let col :usize = x[0].len();

  (0..row)
  .map(|r| {
    (0..col)
    .map( |c| x[r][c]*y[r][c])
    .collect()
  })
  .collect()
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

pub fn mult (x: &Vec<Vec<Num>>, y: &Vec<Vec<Num>>) -> Vec<Vec<Num>> {
  // let ty = transpose(y);
  let ty: Box<Vec<Vec<Num>>> = Box::new(transpose(y));

  x.iter()
  .map(|i| {
    ty.iter()
    .map(|j| dot_product(&i as &Vec<Num>, &j as &Vec<Num>))
    .collect()
  })
  .collect()
}

/*********************************************************************/

pub fn dot_product_par(x: &Vec<Num>, y: &Vec<Num>) -> Num {
  x.par_iter().zip(y.par_iter())
  .map(|t| t.0 * t.1)
  .sum()
}

pub fn sub_par(x: &Vec<Num>, y: &Vec<Num>) -> Vec<Num> {
  x.par_iter().zip(y.par_iter())
  .map(|t| t.0 - t.1)
  .collect()
}
pub fn sub2d_par(x: &Vec<Vec<Num>>, y: &Vec<Vec<Num>>) -> Vec<Vec<Num>>{
  let row :usize = x.len();
  let col :usize = x[0].len();

  (0..row)
  .into_par_iter()
  .map(|r| {
    (0..col)
    // .into_par_iter()
    .map( |c| x[r][c]-y[r][c])
    .collect()
  })
  .collect()
}
pub fn emult_par(x: &Vec<Num>, y: &Vec<Num>) -> Vec<Num> {
  x.par_iter().zip(y.par_iter())
  .map(|t| t.0 * t.1)
  .collect()
}

pub fn emult2d_par(x: &Vec<Vec<Num>>, y: &Vec<Vec<Num>>) -> Vec<Vec<Num>>{
  let row :usize = x.len();
  let col :usize = x[0].len();

  (0..row)
  // .into_par_iter()
  .map(|r| {
    (0..col)
    // .into_par_iter()
    .map( |c| x[r][c]*y[r][c])
    .collect()
  })
  .collect()
}

pub fn transpose_par(x: &Vec<Vec<Num>>) -> Vec<Vec<Num>> {
  //swap_rows_cols(x)

  let row :usize = x.len();
  let col :usize = x[0].len();

  (0..col)
  .into_par_iter()
  .map(|c| {
    (0..row)
    .map( |r| x[r][c] )
    .collect()
  })
  .collect()
}

pub fn mult_par (x: &Vec<Vec<Num>>, y: &Vec<Vec<Num>>) -> Vec<Vec<Num>> {
  let ty = transpose_par(y);

  x.par_iter()
  .map(|i| {
    ty.iter()
    .map(|j| dot_product(&i as &Vec<Num>, &j as &Vec<Num>))
    .collect()
  })
  .collect()
}

fn set_num_threads(n: usize) -> Result<ThreadPool, ThreadPoolBuildError> {
  rayon::ThreadPoolBuilder::new().num_threads(n).build()
}

fn benchmark_rayon_fit(
  n: usize,
  x: &Vec<Vec<Num>>, 
  y: &Vec<Vec<Num>>, 
  alpha: Num,
  iterations: i64) -> Result<f64, ThreadPoolBuildError> {
    {
        match set_num_threads(n) {
          Ok(pool) => pool.install(|| {
            let start_time = time::get_time();
            fit_par(x, y, alpha, iterations);
            let end_time = time::get_time();
            let diffsec = end_time.sec - start_time.sec;   // i64
            let diffsub = end_time.nsec - start_time.nsec; // i32
            let realsec = diffsec as f64 + diffsub as f64 * 1e-9;
            Ok(realsec)
          }),
          Err(e) => Err(e),
        }
    }
}

fn fit(
  x: &Vec<Vec<Num>>, 
  y: &Vec<Vec<Num>>, 
  alpha: Num,
  iterations: i64
  ) -> Vec<Vec<Num>>{

  let tx = transpose(&x);
  let m = y.len() as Num;
  // let (row, col) = (theta.len(), theta[0].len());
  let (row, col) = (x.len(), x[0].len());
  let tmp = alpha/m;
  let a :Vec<Vec<Num>> = vec![vec![tmp; col]; row]; 
  let theta = vec![vec![0.0; 1]; col];

  let ans = (0..iterations)
    .fold( theta, |theta, _iteration|{
     sub2d(&theta, &emult2d(&mult( &tx, &sub2d( &mult( &x, &theta ), &y ) ), &a))
  });

  ans
}

fn fit_par(
  x: &Vec<Vec<Num>>, 
  y: &Vec<Vec<Num>>, 
  alpha: Num,
  iterations: i64
  ) -> Vec<Vec<Num>>{
  
  let tx = transpose(&x);
  let m = y.len() as Num;
  // let (row, col) = (theta.len(), theta[0].len());
  let (row, col) = (x.len(), x[0].len());
  let tmp = alpha/m;
  let a :Vec<Vec<Num>> = vec![vec![tmp; col]; row]; 
  let theta = vec![vec![0.0; 1]; col];

  (0..iterations)
    .fold( theta, |theta, _iteration|{
     sub2d_par(&theta, &emult2d(&mult( &tx, &sub2d( &mult( &x, &theta ), &y ) ), &a))
  })
}

fn nif_fit<'a>(env: Env<'a>, args: &[Term<'a>])-> NifResult<Term<'a>> {
  let pid = env.pid();
  let mut my_env = OwnedEnv::new();

  let saved_list = my_env.run(|env| -> NifResult<SavedTerm> {
    let _x = args[0].in_env(env);
    let _y = args[1].in_env(env);
    let alpha = args[2].in_env(env);
    let iterations = args[3].in_env(env);
    Ok(my_env.save(make_tuple(env, &[_x, _y, alpha, iterations])))
  })?;

  std::thread::spawn(move ||  {
    my_env.send_and_clear(&pid, |env| {
      let result: NifResult<Term> = (|| {
        let tuple = saved_list
        .load(env).decode::<(  
          Term, 
          Term,
          Num,
          i64)>()?; 
        
        let x: Vec<Vec<Num>> = tuple.0.decode()?;
        let y: Vec<Vec<Num>> = tuple.1.decode()?;
        let alpha: Num = tuple.2;
        let iterations: i64 = tuple.3;

        let ans = fit(&x, &y, alpha, iterations);

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

fn rayon_fit<'a>(env: Env<'a>, args: &[Term<'a>])-> NifResult<Term<'a>> {
  let pid = env.pid();
  let mut my_env = OwnedEnv::new();

  let saved_list = my_env.run(|env| -> NifResult<SavedTerm> {
    let _x = args[0].in_env(env);
    let _y = args[1].in_env(env);
    let alpha = args[2].in_env(env);
    let iterations = args[3].in_env(env);
    Ok(my_env.save(make_tuple(env, &[_x, _y, alpha, iterations])))
  })?;

  std::thread::spawn(move ||  {
    my_env.send_and_clear(&pid, |env| {
      let result: NifResult<Term> = (|| {
        let tuple = saved_list
        .load(env).decode::<(
          Term, 
          Term,
          Num,
          i64)>()?; 
        
        let x: Vec<Vec<Num>> = tuple.0.decode()?;
        let y: Vec<Vec<Num>> = tuple.1.decode()?;
        let alpha: Num = tuple.2;
        let iterations: i64 = tuple.3;

        let ans = fit_par(&x, &y, alpha, iterations);

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

fn nif_benchmark<'a>(env: Env<'a>, args: &[Term<'a>])-> NifResult<Term<'a>> {
  let pid = env.pid();
  let mut my_env = OwnedEnv::new();

  let saved_list = my_env.run(|env| -> NifResult<SavedTerm> {
    let _x = args[0].in_env(env);
    let _y = args[1].in_env(env);
    let alpha = args[2].in_env(env);
    let iterations = args[3].in_env(env);
    Ok(my_env.save(make_tuple(env, &[_x, _y, alpha, iterations])))
  })?;

  std::thread::spawn(move ||  {
    my_env.send_and_clear(&pid, |env| {
      let result: NifResult<Term> = (|| {
        let tuple = saved_list
        .load(env).decode::<(
          Term, 
          Term,
          Num,
          i64)>()?; 
        
        let x: Vec<Vec<Num>> = tuple.0.decode()?;
        let y: Vec<Vec<Num>> = tuple.1.decode()?;
        let alpha: Num = tuple.2;
        let iterations: i64 = tuple.3;

        let num = num_cpus::get();
        let mut single:f64 = 0.0;

        println!("header, thread, speedup efficiency");

        (1..=num as usize).collect::<Vec<_>>().iter().for_each(|&n| {
          let minsec = (1..=10).collect::<Vec<_>>().iter().map(|_|
            match benchmark_rayon_fit(n, &x, &y, alpha, iterations) {
              Ok(realsec) => realsec,
              Err(e) => {println!("error: {}", e); ::std::f64::NAN},
            }
          ).fold(0.0/0.0, |m, v| v.min(m));
          match n {
            1 => single = minsec,
            _ => {},
          }
          println!(", {}, {}", n, ((single / minsec / (n as f64)) * 1000.0).round() / 10.0 );
        });

        Ok([0].encode(env))
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
  Ok(dot_product(&x, &y).encode(env))
}

fn nif_sub<'a>(env: Env<'a>, args: &[Term<'a>]) -> NifResult<Term<'a>> {
  let x: Vec<Num> = args[0].decode()?;
  let y: Vec<Num> = args[1].decode()?;
  
  Ok(sub(&x, &y).encode(env))
}

fn nif_emult<'a>(env: Env<'a>, args: &[Term<'a>]) -> NifResult<Term<'a>> {
  let x: Vec<Num> = args[0].decode()?;
  let y: Vec<Num> = args[1].decode()?;
  
  Ok(emult(&x, &y).encode(env))
}

fn new(first: i64, end: i64) -> Vec<i64> {
  (first..end).collect()
}

fn nif_new<'a>(env: Env<'a>, args: &[Term<'a>]) -> NifResult<Term<'a>> {
  let first: i64 = args[0].decode()?;
  let end: i64 = args[1].decode()?;

  Ok(new(first, end).encode(env))
}

fn zeros<'a>(env: Env<'a>, args: &[Term<'a>]) -> NifResult<Term<'a>> {
  let len: usize = (args[0]).decode()?;
  let zero_vec = vec![0; len];
  Ok(zero_vec.encode(env))
}