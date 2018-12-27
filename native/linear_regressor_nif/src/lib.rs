#[macro_use] extern crate rustler;
// #[macro_use] extern crate rustler_codegen;
#[macro_use] extern crate lazy_static;

extern crate num_cpus;

use rustler::{Env, Term, NifResult, Encoder};
use rustler::env::{OwnedEnv, SavedTerm};
use rustler::types::tuple::make_tuple;

mod matrix;

// 関数単位でuseは慣習的ではない(一旦使う)
use matrix::single_thread;
use matrix::multi_thread;
use matrix::single_thread::{dot_product, sub, emult};


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

        let ans = single_thread::fit(&x, &y, alpha, iterations);

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

        let ans = multi_thread::fit_par(&x, &y, alpha, iterations);

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
            match multi_thread::benchmark_rayon_fit(n, &x, &y, alpha, iterations) {
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