#[macro_use] extern crate rustler;
// #[macro_use] extern crate rustler_codegen;
#[macro_use] extern crate lazy_static;

extern crate num_cpus;
extern crate rayon;

mod matrix;
mod ai_ml;
mod atoms;

use matrix::single_core::nif as single;
use matrix::multi_core::nif as multi;
use ai_ml::linear_regressor::nif as lr;

rustler_export_nifs! {
  "Elixir.LinearRegressorNif",
  [
    //("Elixir's func, number of arguments, Rust's func)
    
    ("zeros", 1, single::zeros),
    ("new", 2, single::new),

    // Single Core
    ("dot_product", 2, single::dot_product),
    ("sub", 2, single::sub),
    ("emult", 2, single::emult),
    ("fit", 4, lr::nif_fit),
    
    // Multi Core
    ("rayon_dot_product", 2, multi::dot_product),
    ("rayon_sub", 2, multi::sub),
    ("rayon_emult", 2, multi::emult),
    ("rayon_fit", 4, lr::rayon_fit), 
    
    ("benchmark", 4, lr::benchmark),
  ],
  None
}
