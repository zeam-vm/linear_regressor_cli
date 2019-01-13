pub mod nif;

use matrix::single_core as single;
use matrix::multi_core as multi;

pub fn fit(
  tx: &Vec<Vec<f64>>, 
  ty: &Vec<Vec<f64>>, 
  alpha: f64,
  iterations: i64
  ) -> Vec<Vec<f64>>{

  let x = single::transpose(&tx);
  let y = single::transpose(&ty);
  let m = y.len() as f64;
  // let (row, col) = (theta.len(), theta[0].len());
  let (_row, col) = (x.len(), x[0].len());
  let a = alpha/m;
  // let a :Vec<Vec<f64>> = vec![vec![tmp; col]; row]; 
  let theta = vec![vec![0.0; 1]; col];

  let ans = (0..iterations)
    .fold( theta, |theta, _iteration|{
     single::sub2d(
      &theta, 
      &single::scale(
        &single::mult( 
          &tx, 
          &single::sub2d( 
            &single::mult( &x, &theta ), 
            &y )
           ), 
        a))
  });

  ans
}

pub fn fit_par(
  tx: &Vec<Vec<f64>>, 
  ty: &Vec<Vec<f64>>, 
  alpha: f64,
  iterations: i64
  ) -> Vec<Vec<f64>>{
  
  let x = multi::transpose(&tx);
  let y = multi::transpose(&ty);
  let m = y.len() as f64;
  // let (row, col) = (theta.len(), theta[0].len());
  let (_row, col) = (x.len(), x[0].len());
  let a = alpha/m;
  // let a :Vec<Vec<f64>> = vec![vec![tmp; col]; _row]; 
  let theta = vec![vec![0.0; 1]; col];

  (0..iterations)
    .fold( theta, |theta, _iteration|{
     single::sub2d(
      &theta,  
      &single::scale(
        &multi::mult_y( 
          &tx, 
          &multi::sub2d_xy( 
            &multi::mult_x( &x, &theta ), 
            &y ) 
          ), 
        a))
  })
}