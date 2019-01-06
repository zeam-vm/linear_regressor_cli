pub mod nif;

use matrix::single_core as single;
use matrix::multi_core as multi;

pub fn fit(
  x: &Vec<Vec<f64>>, 
  y: &Vec<Vec<f64>>, 
  alpha: f64,
  iterations: i64
  ) -> Vec<Vec<f64>>{

  let tx = single::transpose(&x);
  let m = y.len() as f64;
  // let (row, col) = (theta.len(), theta[0].len());
  let (row, col) = (x.len(), x[0].len());
  let tmp = alpha/m;
  let a :Vec<Vec<f64>> = vec![vec![tmp; col]; row]; 
  let theta = vec![vec![0.0; 1]; col];

  let ans = (0..iterations)
    .fold( theta, |theta, _iteration|{
     single::sub2d(&theta, &single::emult2d(&single::mult( &tx, &single::sub2d( &single::mult( &x, &theta ), &y ) ), &a))
  });

  ans
}

pub fn fit_par(
  x: &Vec<Vec<f64>>, 
  y: &Vec<Vec<f64>>, 
  alpha: f64,
  iterations: i64
  ) -> Vec<Vec<f64>>{
  
  let tx = single::transpose(&x);
  let m = y.len() as f64;
  // let (row, col) = (theta.len(), theta[0].len());
  let (row, col) = (x.len(), x[0].len());
  let tmp = alpha/m;
  let a :Vec<Vec<f64>> = vec![vec![tmp; col]; row]; 
  let theta = vec![vec![0.0; 1]; col];

  (0..iterations)
    .fold( theta, |theta, _iteration|{
     multi::sub2d_par(&theta, &single::emult2d(&single::mult( &tx, &single::sub2d( &single::mult( &x, &theta ), &y ) ), &a))
  })
}