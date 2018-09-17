#[macro_use] extern crate rustler;
// #[macro_use] extern crate rustler_codegen;
#[macro_use] extern crate lazy_static;



use rustler::{Env, Term, NifResult, Encoder};
use rustler::types::atom::Atom;
use rustler::types::list::ListIterator;
use rustler::Error;

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
        ("add", 2, add),
        ("print_tuple", 1, print_tuple),
        ("sum_list", 1, sum_list),
        ("dot_product", 2, dot_product)
    ],
    None
}

fn add<'a>(env: Env<'a>, args: &[Term<'a>]) -> NifResult<Term<'a>> {
    let num1: i64 = try!(args[0].decode());
    let num2: i64 = try!(args[1].decode());

    Ok((atoms::ok(), num1 + num2).encode(env))
}

fn print_tuple<'a>(env: Env<'a>, args: &[Term<'a>]) -> NifResult<Term<'a>> {
    // タプルの４要素の型定義を行う
    let tuple = (args[0]).decode::<(Atom, f64, i64, String)>()?; 

    println!("Atom: {:?}, Float: {}, Integer: {}, String: {}", tuple.0, tuple.1, tuple.2, tuple.3);

    Ok((atoms::ok(),tuple.3).encode(env)) // {:ok, タプルの４つ目 }を返す
}

fn sum_list<'a>(env: Env<'a>, args: &[Term<'a>]) -> NifResult<Term<'a>> {
    // Gets List
    let iter: ListIterator = try!(args[0].decode());

    let res: Result<Vec<i64>, Error> = iter
        .map(|x| x.decode::<i64>())
        .collect();

    match res {
        Ok(result) => Ok(result.iter().fold(0, |acc, &x| acc + x).encode(env)),
        Err(err) => Err(err),
    }
}

fn dot_product<'a>(env: Env<'a>, args: &[Term<'a>])-> NifResult<Term<'a>> {
    // x : list for 1dim 
    // y : list for 1dim 
    
    // Initialize Arguments
    let x_arg: Term = args[0].in_env(env);
    let y_arg = args[1].in_env(env);

    // Decode to Vector
    let mut x: Vec<i64> = x_arg.decode::<Vec<i64>>()?;
    let y: Vec<i64> = y_arg.decode::<Vec<i64>>()?;
    
    // Calc Dot Product
    let mut ans: i64 = 0; 
    for (i, j) in x.iter_mut().enumerate(){
        //println!("{}: {}", i, (*j)*y[i]);
        ans = ans + (*j)*y[i];
    }

    Ok((atoms::ok(), ans).encode(env))
}

// fn dot_product2(x: Vec<i64>, y:Vec<i64>) -> Vec<i64> {

//     let (lenx, leny) = (x.len(), y.len());

//     loop {
//         match x.next() {
//             Some(x) => {
//                 match y.next() {
//                    Some(y) => {x*y},
//                    None => { break; },
//                 }
//             }, 
//             None => { break },
//         }
//     }
// }

// fn new_list<'a>(env: Env<'a>, _args: &[Term<'a>]) -> NifResult<Term<'a>> {
//     let list = vec![1, 2, 3];
//     Ok(list.encode(env))
// }
