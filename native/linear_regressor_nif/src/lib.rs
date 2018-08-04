#[macro_use] extern crate rustler;
#[macro_use] extern crate rustler_codegen;
#[macro_use] extern crate lazy_static;

extern crate scoped_pool;
extern crate ocl;

use rustler::{Env, Term, NifResult, Encoder};
use rustler::env::{OwnedEnv, SavedTerm};
use rustler::types::tuple::make_tuple;

use ocl::{ProQue, Buffer, MemFlags};

mod atoms {
    rustler_atoms! {
        atom ok;
        //atom error;
        //atom __true__ = "true";
        //atom __false__ = "false";
    }
}

rustler_export_nifs! {
    "Elixir.LinearRegressorNif",
    [
        ("add", 2, add),
        ("fit_nif", 5, fit_nif),
    ],
    None
}

lazy_static! {
    static ref POOL:scoped_pool::Pool = scoped_pool::Pool::new(2);
}

fn add<'a>(env: Env<'a>, args: &[Term<'a>]) -> NifResult<Term<'a>> {
    let num1: i64 = try!(args[0].decode());
    let num2: i64 = try!(args[1].decode());

    Ok((atoms::ok(), num1 + num2).encode(env))
}

pub struct Matrix {
    container: Vec<f64>,
    row_size: usize,
    col_size: usize,
    col_near_pow_2: usize,
    col_shift: usize,
    size: usize,
    transpose: bool,
}



fn near_pow_2(num: usize) -> (usize, usize) {
    if num <= 0 {
        (0, 0)
    } else if (num & (num - 1)) == 0 {
        let mut n = num;
        let mut shift: usize = 0;
        while n > 0 {
            n >>= 1;
            shift += 1;
        }
        (num, shift)
    } else {
        let mut ret: usize = 1;
        let mut n = num;
        let mut shift: usize = 0;
        while n > 0 {
            ret <<= 1;
            n >>= 1;
            shift += 1;
        }
        (ret, shift)
    }
}


impl Matrix {

		pub fn new(row_size: usize, col_size: usize, initial: f64) -> Matrix {
			let col = near_pow_2(col_size + 1 + 1);
			let col_near_pow_2 = col.0.clone();
			let col_shift = col.1.clone();
			let mut size = row_size + 1;
			let mut c = col_shift;
			while c > 0 {
				size <<= 1;
				c -= 1;
			}
			let mut container: Vec<f64> = Vec::with_capacity(size);
			(0..row_size).for_each(|r| {
				print!("r: {}", r);
				(0..col_size).for_each(|c| {
					container.push(initial)
				});
				(col_size..col_near_pow_2).for_each(|_c| {
					container.push(0.0)
				});
			});
      Matrix{
        container: container,
        row_size: row_size,
        col_size: col_size,
        col_near_pow_2: col_near_pow_2,
        col_shift: col_shift,
        size: size,
        transpose: false,
      }
		}

    pub fn new_from_vec(vec: Vec<Vec<f64>>) -> Matrix {
        let row_size: usize = vec.len();
        let col_size = vec[0].len();
        let col = near_pow_2(col_size + 1);
        let col_near_pow_2 = col.0.clone();
        let col_shift = col.1.clone();
        let mut size = row_size;
        let mut c = col_shift;
        while c > 0 {
            size <<= 1;
            c -= 1;
        }
        let mut container: Vec<f64> = Vec::with_capacity(size);
        (0..row_size).for_each(|r| {
            match vec.get(r) {
                Some(vec_r) => {
                    (0..vec_r.len()).for_each(|c| {
                        match vec_r.get(c) {
                            Some(vec_c) => {
                                container.push(*vec_c)
                            },
                            None => {},
                        }
                    });
                    (vec_r.len()..col_near_pow_2).for_each(|_c| {
                        container.push(0.0)
                    });
                },
                None => {},
            }
        });
        Matrix{
            container: container,
            row_size: row_size,
            col_size: col_size,
            col_near_pow_2: col_near_pow_2,
            col_shift: col_shift,
            size: size,
            transpose: false,
        }
    }

    pub fn container(&self) -> Vec<f64> {
        self.container.clone()
    }


    pub fn transpose(&self) -> Matrix {
        Matrix{
            container: self.container(),
            row_size: self.row_size,
            col_size: self.col_size,
            col_near_pow_2: self.col_near_pow_2,
            col_shift: self.col_shift,
            size: self.size,
            transpose: !self.transpose,
        }
    }

    pub fn col_size(&self) -> usize {
        match self.transpose {
            false => self.col_size,
            true => self.row_size
        }
    }

    pub fn row_size(&self) -> usize {
        match self.transpose {
            false => self.row_size,
            true => self.col_size
        }
    }

    pub fn length(&self) -> usize {
        match self.transpose {
            false => self.row_size,
            true => self.col_size
        }
    }

    pub fn size(&self) -> (usize, usize) {
        (self.row_size(), self.col_size())
    }

    pub fn i(&self, row: usize, col: usize) -> usize {
        match self.transpose {
            false => (row << self.col_shift) + col,
            true  => (col << self.col_shift) + row
        }
    }

    pub fn row_vec(&self, row: usize) -> Vec<f64> {
        let mut ret: Vec<f64> = Vec::with_capacity(self.col_size());
        (0..(self.col_size())).for_each(|c| {
            ret.push(self.container[self.i(row, c)]);
        });
        ret
    }

    pub fn col_vec(&self, col: usize) -> Vec<f64> {
        let mut ret: Vec<f64> = Vec::with_capacity(self.row_size());
        (0..(self.row_size())).for_each(|r| {
            ret[r] = self.container[self.i(r, col)];
        });
        ret
    }

    pub fn to_vec(&self) -> Vec<Vec<f64>> {
        let mut ret: Vec<Vec<f64>> = Vec::with_capacity(self.row_size());
        (0..(self.row_size())).for_each(|r| {
            ret.push(self.row_vec(r))
        });
        ret
    }
}

fn fit_nif<'a>(env: Env<'a>, args: &[Term<'a>]) -> NifResult<Term<'a>> {
    let pid = env.pid();
    let mut my_env = OwnedEnv::new();

    let saved_list = my_env.run(|env| -> NifResult<SavedTerm> {
        let x_arg = args[0].in_env(env);
        let y_arg = args[1].in_env(env);
        let theta_arg = args[2].in_env(env);
        let alpha_arg = args[3].in_env(env);
        let iteration_arg = args[4].in_env(env);
        Ok(my_env.save(make_tuple(env, &[x_arg, y_arg, theta_arg, alpha_arg, iteration_arg])))
    })?;

    POOL.spawn(move || {
        my_env.send_and_clear(&pid, |env| {
            let result: NifResult<Term> = (|| {
                let tuple = saved_list.load(env).decode::<(Term, Term, Term, f64, i64)>()?;
                let x = Matrix::new_from_vec(tuple.0.decode::<Vec<Vec<f64>>>()?);
                let y = Matrix::new_from_vec(tuple.1.decode::<Vec<Vec<f64>>>()?);
                let mut theta = Matrix::new_from_vec(tuple.2.decode::<Vec<Vec<f64>>>()?);
                let alpha: f64 = tuple.3;
                let iteration: i64 = tuple.4;
                let m = y.length();
                let tx = x.transpose();
                let size = theta.size();
        				let a = Matrix::new( theta.row_size(), theta.col_size(), alpha * ( 1.0 / m as f64) );
								(0..=iteration).for_each(|_i| {
									let trans_theta = theta.transpose();

								});

                let src = r#"
                __kernel void mult(
                    __global const double* A,
                    __global const double* B,
                    __global double* Result,
                    const int wA, const int sA,
                    const int wB, const int sB) {
                    const int x = get_global_id(0);
                    const int y = get_global_id(1);
                    float value = 0;
                    for (int i = 0; i < wA; ++i) {
                        int index_a = y << sA + i;
                        int index_b = i << sB + x;
                        float elementA = A[index_a];
                        float elementB = B[index_b];
                        value = value + elementA * elementB;
                    }
                    Result[wB * y + x] = value;
                }
                "#;


                let pro_que = ProQue::builder()
                    .src(src)
                    .dims(x.container().capacity()) // TODO: set dims                    .build().expect("Build ProQue");

                Ok(a.to_vec().encode(env))
            })();
            match result {
                Err(_err) => env.error_tuple("test failed".encode(env)),
                Ok(term) => term
            }
        });
    });

    Ok(atoms::ok().encode(env))
}
