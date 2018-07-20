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
                let x = tuple.0.decode::<Vec<Vec<f64>>>()?;
                let y = tuple.1.decode::<Vec<Vec<f64>>>()?;
                let theta = tuple.2.decode::<Vec<Vec<f64>>>()?;
                let alpha: f64 = tuple.3;
                let iteration: i64 = tuple.4;

                Ok(x.encode(env))
            })();
            match result {
                Err(_err) => env.error_tuple("test failed".encode(env)),
                Ok(term) => term
            }
        });
    });

    Ok(atoms::ok().encode(env))
}
