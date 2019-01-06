#[macro_use] extern crate rustler;
// #[macro_use] extern crate rustler_codegen;
#[macro_use] extern crate lazy_static;

extern crate num_cpus;
extern crate rayon;

mod matrix;
mod ai_ml;
mod atoms;

// 関数単位でuseは慣習的ではない(一旦使う)
use matrix::single_core::nif as single;
// use Matrix::MultiCore::nif as mlt;
use ai_ml::linear_regressor::nif as lr;

rustler_export_nifs! {
  "Elixir.LinearRegressorNif",
  [
    //("Elixir's func, number of arguments, Rust's func)
    ("_dot_product", 2, single::dot_product),
    ("_zeros", 1, single::zeros),
    ("_new", 2, single::new),
    ("_sub", 2, single::sub),
    ("_emult", 2, single::emult),
    ("_fit", 4, lr::nif_fit),
    ("_rayon_fit", 4, lr::rayon_fit), 
    ("_nif_benchmark", 4, lr::nif_benchmark),
  ],
  None
}
