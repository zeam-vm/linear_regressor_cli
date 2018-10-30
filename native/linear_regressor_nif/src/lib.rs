#[macro_use] extern crate rustler;
// #[macro_use] extern crate rustler_codegen;
#[macro_use] extern crate lazy_static;

extern crate ocl;
extern crate rayon;
extern crate scoped_pool;

use rustler::{Env, Term, NifResult, Encoder, Error};
use rustler::env::{OwnedEnv, SavedTerm};
// use rustler::types::atom::Atom;
// use rustler::types::list::ListIterator;
// use rustler::types::map::MapIterator;

use rustler::types::tuple::make_tuple;
//use std::mem;
// use std::slice;
 use std::str;
// use std::ops::RangeInclusive;

use rayon::prelude::*;

use ocl::{ProQue, Buffer, MemFlags};

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
    ("_fit", 5, nif_fit), 
    ("_call_ocl", 2, call_ocl),
    ("gpuinfo", 0, gpuinfo),
  ],
  None
}

lazy_static! {
    static ref POOL:scoped_pool::Pool = scoped_pool::Pool::new(2);
}

pub fn dot_product(x: &Vec<Num>, y: &Vec<Num>) -> Num {
  x.iter().zip(y.iter())
  .map(|t| t.0 * t.1)
  .fold(0.0, |sum, i| sum + i)
}

pub fn gpuinfo<'a>(env: Env<'a>, args: &[Term<'a>]) -> NifResult<Term<'a>> {

  use ocl::{Platform};
  let platform = Platform::default();
  println!("PlatformList{:?}", ocl::Platform::list());
  println!("Profile:{:?}", platform.info(ocl::enums::PlatformInfo::Profile));  
  println!("Version:{:?}", platform.info(ocl::enums::PlatformInfo::Version));
  println!("Name:{:?}", platform.info(ocl::enums::PlatformInfo::Name));
  println!("Vendor:{:?}", platform.info(ocl::enums::PlatformInfo::Vendor));
  // println!("Extensions:{:?}", platform.info(ocl::enums::PlatformInfo::Extensions));
  
  Ok(atoms::ok().to_term(env))
}


pub fn call_ocl<'a>(env: Env<'a>, args: &[Term<'a>]) -> NifResult<Term<'a>> {
  let pid = env.pid();
  let mut my_env = OwnedEnv::new();

  let saved_list = my_env.run(|env| -> NifResult<SavedTerm> {
    let _x = args[0].in_env(env);
    let _y = args[1].in_env(env);
    Ok(my_env.save(make_tuple(env, &[_x, _y])))
  })?;


  std::thread::spawn(move ||  {
    my_env.send_and_clear(&pid, |env| {
      let result: NifResult<Term> = (|| {
        let tuple = saved_list
        .load(env).decode::<(
          Term,
          Term
        )>()?; 
        
        let x: Vec<Num> = try!(tuple.0.decode());
        let y: Vec<Num> = try!(tuple.1.decode());
        
        let result: ocl::Result<(Vec<Num>)> = dot_product_ocl(&x, &y);
        match result {
          Ok(res) => Ok(res[0].encode(env)),
          Err(_) =>  Err(Error::BadArg),
        }
      })();
      match result {
          Err(_err) => env.error_tuple("test failed".encode(env)),
          Ok(term) => term
      }  
    });
  });
  Ok(atoms::ok().to_term(env))
}

pub fn dot_product_ocl(x: &Vec<Num>, y: &Vec<Num>) -> ocl::Result<(Vec<Num>)> {
  let src = r#"
    __kernel void calc(
      __global double* x, 
      __global double* y,
      long size, 
      __global double* output
    ) {
      size_t id = get_global_id(0);
      size_t len = size; 

      double tmp = 0;
      
      for(int i = 0; i < len; ++i){
        tmp += x[i + id] * y[i + id];
      }

      output[id] = tmp;
    }
  "#;


  let pro_que = ProQue::builder()
    .src(src)
    .dims(x.len())
    .build().expect("Build ProQue");

  let source_buffer_x = Buffer::builder()
    .queue(pro_que.queue().clone())
    .flags(MemFlags::new().read_only())
    .len(x.len())
    .copy_host_slice(&x)
    .build()?;

  let source_buffer_y = Buffer::builder()
    .queue(pro_que.queue().clone())
    .flags(MemFlags::new().read_only())
    .len(y.len())
    .copy_host_slice(&y)
    .build()?;

  let result_buffer = Buffer::<Num>::builder()
    .queue(pro_que.queue().clone())
    .flags(MemFlags::new().write_only())
    .len(x.len())
    .build()?;

  let kernel = pro_que.kernel_builder("calc")
    .arg(&source_buffer_x)
    .arg(&source_buffer_y)
    .arg(&x.len())
    .arg(&result_buffer)
    .build()?;

  unsafe { kernel.enq()?; }

  let mut result = vec![0.0; result_buffer.len()];
  // let mut result = 0.0;
  result_buffer.read(&mut result).enq()?;
  Ok(result)
}

pub fn sub(x: &Vec<Num>, y: &Vec<Num>) -> Vec<Num> {
  x.iter().zip(y.iter())
  .map(|t| t.0 - t.1)
  .collect()
}

pub fn sub2d(x: &Vec<Vec<Num>>, y: &Vec<Vec<Num>>) -> Vec<Vec<Num>>{
  x.iter().zip(y.iter())
  .map(|t| sub(&t.0.to_vec(), &t.1.to_vec()))
  .collect()
}

pub fn emult(x: &Vec<Num>, y: &Vec<Num>) -> Vec<Num> {
  x.iter().zip(y.iter())
  .map(|t| t.0 * t.1)
  .collect()
}

pub fn emult2d(x: &Vec<Vec<Num>>, y: &Vec<Vec<Num>>) -> Vec<Vec<Num>>{
  x.iter().zip(y.iter())
  .map(|t| emult(&t.0.to_vec(), &t.1.to_vec()))
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
  let ty = transpose(y);

  x.iter()
  .map(|i| {
    ty.iter()
    .map(|j| dot_product(&i.to_vec(), &j.to_vec()))
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
          i64
        )>()?; 
        
        let x: Vec<Vec<Num>> = try!(tuple.0.decode());
        let y: Vec<Vec<Num>> = try!(tuple.1.decode());
        let theta: Vec<Vec<Num>> = try!(tuple.2.decode());
        let alpha: Num = tuple.3;
        let iterations: i64 = tuple.4;

        let tx = transpose(&x);
        let m = y.len() as Num;
        let (row, col) = (theta.len(), theta[0].len());
        let tmp = alpha/m;
        let a :Vec<Vec<Num>> = vec![vec![tmp; col]; row]; 

        let ans = (0..iterations)
          .fold( theta, |theta, _iteration|{
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
  let x: Vec<Num> = args[0].decode()?;
  let y: Vec<Num> = args[1].decode()?;

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
  let first: i64 = try!(args[0].decode());
  let end: i64 = try!(args[1].decode());

  Ok(new(first, end).encode(env))
}

fn zeros<'a>(env: Env<'a>, args: &[Term<'a>]) -> NifResult<Term<'a>> {
  let len: usize = (args[0]).decode()?;
  let zero_vec = vec![0; len];
  Ok(zero_vec.encode(env))
}

