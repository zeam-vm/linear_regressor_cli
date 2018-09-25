#[macro_use] extern crate rustler;
// #[macro_use] extern crate rustler_codegen;
#[macro_use] extern crate lazy_static;



use rustler::{Env, Term, NifResult, Encoder};
// use rustler::env::{OwnedEnv, SavedTerm};
// use rustler::types::atom::Atom;
// use rustler::types::list::ListIterator;
// use rustler::types::map::MapIterator;

// use rustler::types::tuple::make_tuple;
// use std::ops::Range;
// use std::ops::RangeInclusive;

// use rayon::prelude::*;

// type NifResult<T> = Result<T, Error>;

mod atoms {
    rustler_atoms! {
        atom ok;
        //atom error;
        //atom __true__ = "true";
        //atom __false__ = "false";
    }
}

rustler_export_nifs! {
    "Elixir.NifRegressor",
    [
        //("Elixir's func, number of arguments, Rust's func)
        ("nif_dot_product", 2, dot_product),
        ("nif_zeros", 1, zeros),
        ("nif_new", 2, new),
    ],
    None
}

fn new<'a>(env: Env<'a>, args: &[Term<'a>]) -> NifResult<Term<'a>> {
    let first: i64 = try!(args[0].decode());
    let end: i64 = try!(args[1].decode());
    
    let vec = (first..end).collect::<Vec<_>>();

    Ok((atoms::ok(), vec).encode(env))
}

// fn to_range(arg: Term) -> Result<RangeInclusive<f64>, Error> {
//     let vec:Vec<(Term, Term)> = arg.decode::<MapIterator>()?.collect();
//     match (&*vec[0].0.atom_to_string()?, &*vec[0].1.atom_to_string()?) {
//         ("__struct__", "Elixir.Range") => {
//             let first = vec[1].1.decode::<f64>()?;
//             let last = vec[2].1.decode::<f64>()?;
//             Ok(first ..= last)
//         },
//         _ => Err(Error::BadArg),
//     }
// }

// fn to_list(arg: Term) -> Result<Vec<f64>, Error> {
//     match (arg.is_map(), arg.is_list() || arg.is_empty_list()) {
//         (true, false) => Ok(to_range(arg)?.collect::<Vec<f64>>()),
//         (false, true) => Ok(arg.decode::<Vec<f64>>()?),
//         _ => Err(Error::BadArg),
//     }
// }

fn dot_product<'a>(env: Env<'a>, args: &[Term<'a>])-> NifResult<Term<'a>> {
    // x : list for 1dim 
    // y : list for 1dim 
    
    // Initialize Arguments
    // Decode to Vector
    let x: Vec<f64> = args[0].decode()?;
    let y: Vec<f64> = args[1].decode()?;
    
    // Main Process 
    let tuple = x.iter().zip(y.iter());
    let mut ans = 0.0;
    for (i, j) in tuple{
        ans = ans + (i*j);
    }

    // Return
    Ok((atoms::ok(), ans).encode(env))
}

// fn dot_product2<'a>(env: Env<'a>, args: &[Term<'a>])-> NifResult<Term<'a>> {
//     // x : list for 1dim 
//     // y : list for 1dim 

//     // Initialize Arguments
//     // let x_arg: Term = args[0].in_env(env);
//     // let y_arg: Term = args[1].in_env(env);

//     // Decode to Vector
//     let x_arg = args[0].decode()?;
//     let y_arg = args[1].decode()?;

//     match (to_list(x_arg), to_list(y_arg)) {
//         (Ok(x), Ok(y)) => {
//             let tuple = x.iter().zip(y.iter());
//             let mut ans = 0.0;
//             for (i, j) in tuple{
//                 ans = ans + (i*j);
//             }
//             Ok((atoms::ok(), ans).encode(env))
//         },
//         (Err(err1), Err(err2)) => Err(err1),
//     }

//     // Calc Dot Product
//     // let mut ans: i64 = 0; 
//     // for (i, j) in x.iter_mut().enumerate(){
//     //     //println!("{}: {}", i, (*j)*y[i]);
//     //     ans = ans + (*j)*y[i];
//     // }


// }

fn zeros<'a>(env: Env<'a>, args: &[Term<'a>]) -> NifResult<Term<'a>> {
    let len = (args[0]).decode::<(usize)>()?;
    let zero_vec = vec![0; len];
    Ok(zero_vec.encode(env))
}