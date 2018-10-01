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

// macro_rules! attach_nif {
//     ( $i:item, $x:expr ) => ("item", x, item)
// }

rustler_export_nifs! {
    "Elixir.NifRegressor",
    [
        //("Elixir's func, number of arguments, Rust's func)
        ("dot_product", 2, dot_product),
        ("nif_zeros", 1, zeros),
        ("nif_new", 2, new),
        ("sub", 2, sub)
    ],
    None
}

fn dot_product<'a>(env: Env<'a>, args: &[Term<'a>])-> NifResult<Term<'a>> {
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

fn sub<'a>(env: Env<'a>, args: &[Term<'a>]) -> NifResult<Term<'a>> {
    let x: Vec<f64> = args[0].decode()?;
    let y: Vec<f64> = args[1].decode()?;
    
    let ans: Vec<f64> = x.iter().zip(y.iter())
        .map(|t| t.0 - t.1)
        .collect();
    
    Ok((atoms::ok(), ans).encode(env))
}

// Multiplication for each elements of vector
fn emult<'a>(env: Env<'a>, args: &[Term<'a>]) -> NifResult<Term<'a>> {
    let x: Vec<f64> = args[0].decode()?;
    let y: Vec<f64> = args[1].decode()?;
    
    let ans: Vec<f64> = x.iter().zip(y.iter())
        .map(|t| t.0 * t.1)
        .collect();
    
    Ok((atoms::ok(), ans).encode(env))
}

fn new<'a>(env: Env<'a>, args: &[Term<'a>]) -> NifResult<Term<'a>> {
    let first: i64 = try!(args[0].decode());
    let end: i64 = try!(args[1].decode());
    
    let vec : Vec<i64> = (first..end).collect();

    Ok((atoms::ok(), vec).encode(env))
}

fn zeros<'a>(env: Env<'a>, args: &[Term<'a>]) -> NifResult<Term<'a>> {
    let len = (args[0]).decode::<(usize)>()?;
    let zero_vec = vec![0; len];
    Ok(zero_vec.encode(env))
}