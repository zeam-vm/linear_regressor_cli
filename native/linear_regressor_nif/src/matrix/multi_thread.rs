extern crate rayon;
extern crate time;

use matrix::multi_thread::rayon::prelude::*;
use matrix::multi_thread::rayon::ThreadPool;
use matrix::multi_thread::rayon::ThreadPoolBuildError;
use matrix::single_thread::{transpose, dot_product, mult, sub2d, emult2d};

pub fn dot_product_par(x: &Vec<f64>, y: &Vec<f64>) -> f64 {
  x.par_iter().zip(y.par_iter())
  .map(|t| t.0 * t.1)
  .sum()
}

pub fn sub_par(x: &Vec<f64>, y: &Vec<f64>) -> Vec<f64> {
  x.par_iter().zip(y.par_iter())
  .map(|t| t.0 - t.1)
  .collect()
}

pub fn sub2d_par(x: &Vec<Vec<f64>>, y: &Vec<Vec<f64>>) -> Vec<Vec<f64>>{
  let row :usize = x.len();
  let col :usize = x[0].len();

  (0..row)
  // .into_par_iter()
  .map(|r| {
    (0..col)
    // .into_par_iter()
    .map( |c| x[r][c]-y[r][c])
    .collect()
  })
  .collect()
}
pub fn emult_par(x: &Vec<f64>, y: &Vec<f64>) -> Vec<f64> {
  x.par_iter().zip(y.par_iter())
  .map(|t| t.0 * t.1)
  .collect()
}

pub fn emult2d_par(x: &Vec<Vec<f64>>, y: &Vec<Vec<f64>>) -> Vec<Vec<f64>>{
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

pub fn transpose_par(x: &Vec<Vec<f64>>) -> Vec<Vec<f64>> {
  //swap_rows_cols(x)

  let row :usize = x.len();
  let col :usize = x[0].len();

  (0..col)
  // .into_par_iter()
  .map(|c| {
    (0..row)
    .map( |r| x[r][c] )
    .collect()
  })
  .collect()
}

pub fn mult_par (x: &Vec<Vec<f64>>, y: &Vec<Vec<f64>>) -> Vec<Vec<f64>> {
  let ty = transpose_par(y);

  x.par_iter()
  .map(|i| {
    ty.iter()
    .map(|j| dot_product(&i as &Vec<f64>, &j as &Vec<f64>))
    .collect()
  })
  .collect()
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

pub fn fit_par(
  x: &Vec<Vec<f64>>, 
  y: &Vec<Vec<f64>>, 
  alpha: f64,
  iterations: i64
  ) -> Vec<Vec<f64>>{
  
  let tx = transpose(&x);
  let m = y.len() as f64;
  // let (row, col) = (theta.len(), theta[0].len());
  let (row, col) = (x.len(), x[0].len());
  let tmp = alpha/m;
  let a :Vec<Vec<f64>> = vec![vec![tmp; col]; row]; 
  let theta = vec![vec![0.0; 1]; col];

  (0..iterations)
    .fold( theta, |theta, _iteration|{
     sub2d_par(&theta, &emult2d(&mult( &tx, &sub2d( &mult( &x, &theta ), &y ) ), &a))
  })
}

