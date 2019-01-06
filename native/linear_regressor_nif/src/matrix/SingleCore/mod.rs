pub mod nif;

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

fn new(first: i64, end: i64) -> Vec<i64> {
  (first..end).collect()
}