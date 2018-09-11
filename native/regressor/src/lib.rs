#[macro_use] extern crate rustler;
#[macro_use] extern crate rustler_codegen;
#[macro_use] extern crate lazy_static;

use rustler::{Env, Term, NifResult, Encoder};
use rustler::types::atom::Atom;
use rustler::types::list::ListIterator;
use rustler::Error;

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
        ("add", 2, add),
        ("print_tuple", 1, print_tuple)
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

