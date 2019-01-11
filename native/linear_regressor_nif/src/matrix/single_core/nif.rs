use rustler::{Env, Term, NifResult, Encoder};
use matrix::single_core as sc;

use rustler::env::{OwnedEnv, SavedTerm};
use rustler::types::tuple::make_tuple;
use atoms;

pub fn dot_product<'a>(env: Env<'a>, args: &[Term<'a>])-> NifResult<Term<'a>> {
  let x: Vec<f64> = args[0].decode()?;
  let y: Vec<f64> = args[1].decode()?;

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
  let pid = env.pid();
  let mut my_env = OwnedEnv::new();

  let saved_list = my_env.run(|env| -> NifResult<SavedTerm> {
    let _a = args[0].in_env(env);
    let _b = args[1].in_env(env);
    Ok(my_env.save(make_tuple(env, &[_a, _b])))
  })?;

  std::thread::spawn(move ||  {
    my_env.send_and_clear(&pid, |env| {
      let result: NifResult<Term> = (|| {
        let tuple = saved_list
        .load(env).decode::<(  
          i64,
          i64)>()?; 
        
        let first: i64 = tuple.0;
        let end: i64 = tuple.1;

        let ans = sc::new(first, end);

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

pub fn zeros<'a>(env: Env<'a>, args: &[Term<'a>]) -> NifResult<Term<'a>> {
  let len: usize = (args[0]).decode()?;
  let zero_vec = vec![0; len];
  Ok(zero_vec.encode(env))
}