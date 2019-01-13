extern crate time;

use rustler::{Env, Term, NifResult, Encoder};
use rustler::env::{OwnedEnv, SavedTerm};
use rustler::types::tuple::make_tuple;
use rayon::ThreadPoolBuildError;
use rayon::ThreadPool;
use atoms;

use ai_ml::linear_regressor as lr;
use std::time::{Instant, Duration};

macro_rules! measure {
  ( $x:expr) => {
    {
      let start = Instant::now();
      let result = $x;
      let end = start.elapsed();
      (result, end)
    }
  };
}

// lazy_static! {
//   static ref POOL:scoped_pool::Pool = scoped_pool::Pool::new(8);
// }

pub fn nif_fit<'a>(env: Env<'a>, args: &[Term<'a>])-> NifResult<Term<'a>> {
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
          f64,
          i64)>()?; 
        
        let tx: Vec<Vec<f64>> = tuple.0.decode()?;
        let ty: Vec<Vec<f64>> = tuple.1.decode()?;
        let alpha: f64 = tuple.2;
        let iterations: i64 = tuple.3;

        let mut fit_time :Duration = Duration::new(0, 0);
        let (ans, fit_ot) = measure!({
          lr::fit(&tx, &ty, alpha, iterations)
        });
        fit_time = fit_time + fit_ot;
        println!("fit_time:{:?}", fit_time);

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

pub fn rayon_fit<'a>(env: Env<'a>, args: &[Term<'a>])-> NifResult<Term<'a>> {
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
          f64,
          i64)>()?; 
        
        let tx: Vec<Vec<f64>> = tuple.0.decode()?;
        let ty: Vec<Vec<f64>> = tuple.1.decode()?;
        let alpha: f64 = tuple.2;
        let iterations: i64 = tuple.3;

        let mut fit_time :Duration = Duration::new(0, 0);

        let (ans, fit_ot) = measure!({
          lr::fit_par(&tx, &ty, alpha, iterations)
        });
        fit_time = fit_time + fit_ot;
        println!("fit_time:{:?}", fit_time);
        // println!("theta:{:?}", ans);

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

pub fn benchmark<'a>(env: Env<'a>, args: &[Term<'a>])-> NifResult<Term<'a>> {
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
          f64,
          i64)>()?; 
        
        let x: Vec<Vec<f64>> = tuple.0.decode()?;
        let y: Vec<Vec<f64>> = tuple.1.decode()?;
        let alpha: f64 = tuple.2;
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

fn set_num_threads(n: usize) -> Result<ThreadPool, ThreadPoolBuildError> {
  rayon::ThreadPoolBuilder::new().num_threads(n).build()
}

pub fn benchmark_rayon_fit(
  n: usize,
  x: &Vec<Vec<f64>>, 
  y: &Vec<Vec<f64>>, 
  alpha: f64,
  iterations: i64) -> Result<f64, ThreadPoolBuildError> {
    {
        match set_num_threads(n) {
          Ok(pool) => pool.install(|| {
            let start_time = time::get_time();
            lr::fit_par(x, y, alpha, iterations);
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
