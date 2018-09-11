#[macro_use] extern crate rustler;
// #[macro_use] extern crate rustler_codegen;
#[macro_use] extern crate lazy_static;

extern crate ocl;
extern crate rayon;
extern crate scoped_pool;

use rustler::{Env, Term, NifResult, Encoder, Error};
use rustler::env::{OwnedEnv, SavedTerm};
use rustler::types::map::MapIterator;
use rustler::types::binary::Binary;

use rustler::types::tuple::make_tuple;
use std::mem;
use std::slice;
use std::str;
use std::ops::RangeInclusive;

use rayon::prelude::*;

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
        ("dot_product", 2, dot_product),
        ("fit_nif", 5, fit_nif),
    ],
    None
}


fn dot_product


