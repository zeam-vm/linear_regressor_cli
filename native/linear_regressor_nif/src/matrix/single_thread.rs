pub fn dot_product(x: &Vec<f64>, y: &Vec<f64>) -> f64 {
  // x.iter().zip(y.iter())
  // .map(|t| t.0 * t.1)
  // .fold(0.0, |sum, i| sum + i)

  let row :usize = x.len();
  
  (0..row)
  .map(|r| x[r]*y[r] )
  .fold(0.0, |sum, i| sum + i)
}

pub fn sub(x: &Vec<f64>, y: &Vec<f64>) -> Vec<f64> {
  // x.iter().zip(y.iter())
  // .map(|t| t.0 - t.1)
  // .collect()
  let row :usize = x.len();
  
  (0..row)
  .map(|r| x[r]-y[r] )
  .collect()
}

pub fn sub2d(x: &Vec<Vec<f64>>, y: &Vec<Vec<f64>>) -> Vec<Vec<f64>>{
  // x.iter().zip(y.iter())
  // .map(|t| sub(&t.0 as &Vec<f64>, &t.1 as &Vec<f64>))
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

pub fn emult(x: &Vec<f64>, y: &Vec<f64>) -> Vec<f64> {
  // x.iter().zip(y.iter())
  // .map(|t| t.0 * t.1)
  // .collect()

  let row :usize = x.len();
  
  (0..row)
  .map(|r| x[r]*y[r] )
  .collect()
}

pub fn emult2d(x: &Vec<Vec<f64>>, y: &Vec<Vec<f64>>) -> Vec<Vec<f64>>{
  // x.iter().zip(y.iter())
  // .map(|t| emult(&t.0 as &Vec<f64>, &t.1 as &Vec<f64>))
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

pub fn transpose(x: &Vec<Vec<f64>>) -> Vec<Vec<f64>> {
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

pub fn mult (x: &Vec<Vec<f64>>, y: &Vec<Vec<f64>>) -> Vec<Vec<f64>> {
  // let ty = transpose(y);
  let ty: Box<Vec<Vec<f64>>> = Box::new(transpose(y));

  x.iter()
  .map(|i| {
    ty.iter()
    .map(|j| dot_product(&i as &Vec<f64>, &j as &Vec<f64>))
    .collect()
  })
  .collect()
}

pub fn fit(
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

  let ans = (0..iterations)
    .fold( theta, |theta, _iteration|{
     sub2d(&theta, &emult2d(&mult( &tx, &sub2d( &mult( &x, &theta ), &y ) ), &a))
  });

  ans
}