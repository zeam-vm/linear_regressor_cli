#[macro_use] extern crate rustler;
#[macro_use] extern crate rustler_codegen;
#[macro_use] extern crate lazy_static;

extern crate scoped_pool;

use rustler::{Env, Term, NifResult, Encoder};
use rustler::env::{OwnedEnv, SavedTerm};
use rustler::types::tuple::make_tuple;

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
    pub container: Vec<f64>,
    pub row_max: usize,
    pub col_max: usize,
    pub col_size: usize,
    pub col_shift: usize,
    pub size: usize,
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

    pub fn new(vec: Vec<Vec<f64>>) -> Matrix {
        let row_max: usize = vec.len();
        let col_max = vec[0].len();
        let col = near_pow_2(col_max + 1);
        let col_size = col.0.clone();
        let col_shift = col.1.clone();
        let mut size = row_max;
        let mut c = col_shift;
        while c > 0 {
            size <<= 1;
            c -= 1;
        }
        let mut container: Vec<f64> = Vec::with_capacity(size);
        (0..row_max).for_each(|r| {
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
                    (vec_r.len()..col_size).for_each(|_c| {
                        container.push(0.0)
                    });
                },
                None => {},
            }
        });
        Matrix{container: container,
                row_max: row_max,
                col_max: col_max,
                col_size: col_size,
                col_shift: col_shift,
                size: size,}
    }

    pub fn i(&self, row: usize, col: usize) -> usize {
        (row << self.col_shift) + col
    }

    pub fn t(&self, row: usize, col: usize) -> usize {
        self.i(col, row)
    }

    pub fn row_vec(&self, row: usize) -> Vec<f64> {
        let mut ret: Vec<f64> = Vec::with_capacity(self.col_max);
        (0..(self.col_max)).for_each(|c| {
            ret.push(self.container[self.i(row, c)]);
        });
        ret
    }

    pub fn col_vec(&self, col: usize) -> Vec<f64> {
        let mut ret: Vec<f64> = Vec::with_capacity(self.row_max);
        (0..(self.row_max)).for_each(|r| {
            ret[r] = self.container[self.i(r, col)];
        });
        ret
    }

    pub fn to_vec(&self) -> Vec<Vec<f64>> {
        let mut ret: Vec<Vec<f64>> = Vec::with_capacity(self.row_max);
        (0..(self.row_max)).for_each(|r| {
            ret.push(self.row_vec(r))
        });
        ret
    }

	pub fn length(&self) -> usize {
		self.row_max
	}
}

fn sub(x: &Matrix, y: &Matrix) -> Matrix {
	let row_max = x.row_max;
	let col_max = x.col_max;
	let col_size = x.col_size;
    let col_shift = x.col_shift;
    let size = x.size;
    let mut container: Vec<f64> = Vec::with_capacity(size);
    (0..size).for_each(|i| {
    	container.push(x.container[i] - y.container[i]);
    });
    Matrix{container: container,
        row_max: row_max,
        col_max: col_max,
        col_size: col_size,
        col_shift: col_shift,
        size: size,}
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
                let x = Matrix::new(tuple.0.decode::<Vec<Vec<f64>>>()?);
                let y = Matrix::new(tuple.1.decode::<Vec<Vec<f64>>>()?);
                let theta = Matrix::new(tuple.2.decode::<Vec<Vec<f64>>>()?);
                let alpha: f64 = tuple.3;
                let iteration: i64 = tuple.4;
                let z = y.length();
                //Ok(tuple.0)
                Ok(z.encode(env))
            })();
            match result {
                Err(_err) => env.error_tuple("test failed".encode(env)),
                Ok(term) => term
            }
        });
    });

    Ok(atoms::ok().encode(env))
}
