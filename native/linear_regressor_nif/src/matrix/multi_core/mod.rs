#![allow(dead_code)]

pub mod nif;

extern crate rayon;
use matrix::multi_core::rayon::prelude::*;
// use Matrix::MultiCore::rayon::ThreadPool;
// use Matrix::MultiCore::rayon::ThreadPoolBuildError;

use matrix::single_core as sc;

pub fn dot_product(x: &Vec<f64>, y: &Vec<f64>) -> f64 {
  x.par_iter().zip(y.par_iter())
  .map(|t| t.0 * t.1)
  .sum()
}

pub fn sub(x: &Vec<f64>, y: &Vec<f64>) -> Vec<f64> {
  x.iter().zip(y.iter())
  .map(|t| t.0 - t.1)
  .collect()
}

pub fn sub2d_xy(x: &Vec<Vec<f64>>, y: &Vec<Vec<f64>>) -> Vec<Vec<f64>>{
  x.par_iter().zip(y.par_iter())
  .map(|t| sub(&t.0 as &Vec<f64>, &t.1 as &Vec<f64>))
  .collect()
}

pub fn sub2d_x(x: &Vec<Vec<f64>>, y: &Vec<Vec<f64>>) -> Vec<Vec<f64>>{
  let row :usize = x.len();
  let col :usize = x[0].len();

  (0..row)
  .into_par_iter()
  .map(|r| {
    (0..col)
    .map( |c| x[r][c]-y[r][c])
    .collect()
  })
  .collect()
}

pub fn sub2d_y(x: &Vec<Vec<f64>>, y: &Vec<Vec<f64>>) -> Vec<Vec<f64>>{
  let row :usize = x.len();
  let col :usize = x[0].len();

  (0..row)
  .map(|r| {
    (0..col)
    .into_par_iter()
    .map( |c| x[r][c]-y[r][c])
    .collect()
  })
  .collect()
}

pub fn scale (x: &Vec<Vec<f64>>, y: f64) -> Vec<Vec<f64>> {
  x.par_iter()
  .map(|r| {
    r.par_iter()
    .map(|c| c*y)
    .collect()
  })
  .collect()
}

pub fn emult(x: &Vec<f64>, y: &Vec<f64>) -> Vec<f64> {
  x.par_iter().zip(y.par_iter())
  .map(|t| t.0 * t.1)
  .collect()
}

pub fn emult2d(x: &Vec<Vec<f64>>, y: &Vec<Vec<f64>>) -> Vec<Vec<f64>>{
  let row :usize = x.len();
  let col :usize = x[0].len();

  (0..row)
  .into_par_iter()
  .map(|r| {
    (0..col)
    .into_par_iter()
    .map( |c| x[r][c]*y[r][c])
    .collect()
  })
  .collect()
}

pub fn transpose(x: &Vec<Vec<f64>>) -> Vec<Vec<f64>> {
  //swap_rows_cols(x)

  let row :usize = x.len();
  let col :usize = x[0].len();

  (0..col)
  .into_par_iter()
  .map(|c| {
    (0..row)
    .into_par_iter()
    .map( |r| x[r][c] )
    .collect()
  })
  .collect()
}

pub fn mult_x(x: &Vec<Vec<f64>>, y: &Vec<Vec<f64>>) -> Vec<Vec<f64>> {
  let ty = sc::transpose(y);

  x.par_iter()
  .map(|i| {
    ty.iter()
    .map(|j| sc::dot_product(&i as &Vec<f64>, &j as &Vec<f64>))
    .collect()
  })
  .collect()
}


pub fn mult_y(x: &Vec<Vec<f64>>, y: &Vec<Vec<f64>>) -> Vec<Vec<f64>> {
  let ty = sc::transpose(y);

  x.iter()
  .map(|i| {
    ty.par_iter()
    .map(|j| sc::dot_product(&i as &Vec<f64>, &j as &Vec<f64>))
    .collect()
  })
  .collect()
}


pub fn mult_xy(x: &Vec<Vec<f64>>, y: &Vec<Vec<f64>>) -> Vec<Vec<f64>> {
  let ty = sc::transpose(y);

  x.par_iter()
  .map(|i| {
    ty.par_iter()
    .map(|j| sc::dot_product(&i as &Vec<f64>, &j as &Vec<f64>))
    .collect()
  })
  .collect()
}