pub mod nif;

extern crate rayon;
use Matrix::MultiCore::rayon::prelude::*;
// use Matrix::MultiCore::rayon::ThreadPool;
// use Matrix::MultiCore::rayon::ThreadPoolBuildError;

use Matrix::SingleCore as sc;

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
    .map(|j| sc::dot_product(&i as &Vec<f64>, &j as &Vec<f64>))
    .collect()
  })
  .collect()
}