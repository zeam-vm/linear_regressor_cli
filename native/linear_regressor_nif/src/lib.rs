#[macro_use] extern crate rustler;
// #[macro_use] extern crate rustler_codegen;
#[macro_use] extern crate lazy_static;

extern crate ocl;
extern crate rayon;
extern crate scoped_pool;

use rustler::{Env, Term, NifResult, Encoder};
use rustler::env::{OwnedEnv, SavedTerm};
use rustler::types::tuple::make_tuple;
use std::str;

use ocl::{ProQue, Buffer, MemFlags};

type Num = f64;

mod atoms {
  rustler_atoms! {
    atom ok;
    // atom error;
    //atom __true__ = "true";
    //atom __false__ = "false";
  }
}

rustler_export_nifs! {
  "Elixir.LinearRegressorNif",
  [
    //("Elixir's func, number of arguments, Rust's func)
    ("_dot_product", 2, nif_dot_product),
    ("_zeros", 1, zeros),
    ("_new", 2, nif_new),
    ("_sub", 2, nif_sub),
    ("_emult", 2, nif_emult),
    ("_fit", 5, nif_fit), 
    ("_call_ocl", 2, call_ocl),
    ("gpuinfo", 0, gpuinfo),
  ],
  None
}

// lazy_static! {
//     static ref POOL:scoped_pool::Pool = scoped_pool::Pool::new(2);
// }

pub fn dot_product(x: &Vec<Num>, y: &Vec<Num>) -> Num {
  x.iter().zip(y.iter())
  .map(|t| t.0 * t.1)
  .fold(0.0, |sum, i| sum + i)
}

pub fn gpuinfo<'a>(env: Env<'a>, _args: &[Term<'a>]) -> NifResult<Term<'a>> {
  use ocl::{Platform, Device};
  use ocl::enums::{PlatformInfo, DeviceInfo, DeviceInfoResult};

  let platform = Platform::default();
  let device = Device::first(platform).unwrap();

  println!("PlatformList{:?}", Platform::list());
  println!("Profile:{:?}", platform.info(PlatformInfo::Profile));  
  println!("Version:{:?}", platform.info(PlatformInfo::Version));
  println!("Name:{:?}", platform.info(PlatformInfo::Name));
  println!("Vendor:{:?}", platform.info(PlatformInfo::Vendor));
  // println!("Extensions:{:?}", platform.info(ocl::enums::PlatformInfo::Extensions));
  println!("Device Name:{:?}", device.name());
  println!("Device Vendor:{:?}", device.vendor());

  // let max_local_size = match device.info(MaxWorkGroupSize){ 
  //   //OclResult<DeviceInfoResult>
  //   Ok(res) => res,
  //   Err(err) => err
  // };

  let max_local_size: usize = match device.info(DeviceInfo::MaxWorkGroupSize){
    Ok(DeviceInfoResult::MaxWorkGroupSize(res)) => res,
    _ => { 
      println!("failed to get DeviceInfoResult::MaxWorkGroupSize");
      0
    },
  };

  // assert_eq!(max_local_size, 1024);
  println!("max_local_size:{:?}", max_local_size);
  
  Ok(atoms::ok().to_term(env))
}


pub fn call_ocl<'a>(env: Env<'a>, args: &[Term<'a>]) -> NifResult<Term<'a>> {
  use rustler::Error;

  let pid = env.pid();
  let mut my_env = OwnedEnv::new();

  let saved_list = my_env.run(|env| -> NifResult<SavedTerm> {
    let _x = args[0].in_env(env);
    let _y = args[1].in_env(env);
    Ok(my_env.save(make_tuple(env, &[_x, _y])))
  })?;

  std::thread::spawn(move ||  {
    my_env.send_and_clear(&pid, |env| {
      let result: NifResult<Term> = (|| {
        let tuple = saved_list
        .load(env).decode::<(
          Term,
          Term
        )>()?; 
        
        let x: Vec<Num> = try!(tuple.0.decode());
        let y: Vec<Num> = try!(tuple.1.decode());

        let result: ocl::Result<(Vec<Num>)> = dot_product_ocl(&x, &y);
        match result {
          Ok(res) => {
            // println!("{:?}", res);
            Ok(res.iter().fold(0.0, |sum, i| sum + i).encode(env))
          },
          Err(_) =>  Err(Error::BadArg),
        }
      })();
      match result {
          Err(_err) => env.error_tuple("test failed".encode(env)),
          Ok(term) => term
      }  
    });
  });
  Ok(atoms::ok().to_term(env))
}

pub fn dot_product_ocl(x: &Vec<Num>, y: &Vec<Num>) -> ocl::Result<(Vec<Num>)> {
  use ocl::{Platform, Device};
  use ocl::enums::{DeviceInfo, DeviceInfoResult};

  let vec_size = x.len();
  let platform = Platform::default();
  let device = Device::first(platform).unwrap(); 
  
  //ワークグループ総数
  let max_work_group_size: usize = match device.info(DeviceInfo::MaxWorkGroupSize){
    Ok(DeviceInfoResult::MaxWorkGroupSize(res)) => res,
    _ => { 
      println!("failed to get DeviceInfoResult::MaxWorkGroupSize");
      1
    },
  };

  // グローバルワークアイテムの総数 ? ワークグループあたりのワークアイテム数？
  // let max_work_item_size: Vec<usize> = match device.info(DeviceInfo::MaxWorkItemSizes){
  //   Ok(DeviceInfoResult::MaxWorkItemSizes(res)) => res,
  //   _ => { 
  //     println!("failed to get DeviceInfoResult::MaxWorkGroupSize");
  //     vec![1; 3]
  //   },
  // };

  let kernel_vector_size = 4;

  let local_work_item_size = 1; // if changed on MBA, don't execute 
  let global_work_item_size = vec_size / kernel_vector_size; // because using double4
  let num_groups = match global_work_item_size < local_work_item_size{
    false => global_work_item_size / local_work_item_size,
    true => 1,
  };
  
  // println!("max_work_group_size:{:?}", max_work_group_size);
  // println!("max_work_item_size:{:?}", max_work_item_size);
  // println!("vec_size:{:?}", vec_size);
  // println!("local_size:{:?}", local_work_item_size);
  // println!("global_size:{:?}", global_work_item_size);
  // println!("..so double4 * {:?} exsits", global_work_item_size);
  // println!("num_groups:{:?}", num_groups);

  let src = r#"
    __kernel void calc(
      __global double4* x, 
      __global double4* y,
      __global double* output,
      __local double4* partial_sums
    ){
      //printf("Hello world");

      int lid = get_local_id(0);
      int gid = get_global_id(0);
      int offset = get_local_size(0);

      // printf("group_size:%d\n", offset);
      // printf("x*y[%d] = %lf\n", gid, x[gid]*y[gid]);

      partial_sums[lid] = x[gid] * y[gid];
      barrier(CLK_LOCAL_MEM_FENCE);

      // printf("partial_sums[%d]= %lf\n", lid, partial_sums[lid]);

      for(int i = offset >> 1; i > 0; i >>= 1) {
          if(lid < i) {
              partial_sums[lid] += partial_sums[lid + i];
          }
          barrier(CLK_LOCAL_MEM_FENCE);
      }

      // printf("group_id[%d]\n", get_group_id(0));

      if(lid == 0) {
        output[get_group_id(0)] = dot(partial_sums[0], (double4)1.0);
      }
    }
  "#;

  let pro_que = ProQue::builder()
    .src(src)
    .dims(max_work_group_size)
    .build().expect("Build ProQue");

  let source_buffer_x = Buffer::builder()
    .queue(pro_que.queue().clone())
    .flags(MemFlags::new().read_write())
    .len(vec_size)
    .copy_host_slice(&x)
    .build()?;

  let source_buffer_y = Buffer::builder()
    .queue(pro_que.queue().clone())
    .flags(MemFlags::new().read_write())
    .len(vec_size)
    .copy_host_slice(&y)
    .build()?;

  let output_buffer = Buffer::<Num>::builder()
    .queue(pro_que.queue().clone())
    .flags(MemFlags::new().write_only())
    .len(num_groups)
    .fill_val(0f64)
    .build()?;

  let kernel : ocl::Kernel = pro_que.kernel_builder("calc")
    .arg(&source_buffer_x) 
    .arg(&source_buffer_y)
    .arg(&output_buffer)
    .arg_local::<Num>(local_work_item_size)
    .build()?;

  // let _res = pro_que.queue().flush();

  unsafe { 
    kernel.cmd()
      .queue(pro_que.queue())
      .global_work_offset(kernel.default_global_work_offset())
      .global_work_size(global_work_item_size)
      // .local_work_size(kernel.default_local_work_size())
      .local_work_size(local_work_item_size)
      .enq()?;
  }

  //unsafe{ kernel.enq()?; }

  println!("success execute kernel code");

  
  let mut result = vec![0f64; num_groups];
  output_buffer.read(&mut result).enq()?;

  Ok(result)
  // Ok(atoms::ok().encode(env))
}

pub fn sub(x: &Vec<Num>, y: &Vec<Num>) -> Vec<Num> {
  x.iter().zip(y.iter())
  .map(|t| t.0 - t.1)
  .collect()
}

pub fn sub2d(x: &Vec<Vec<Num>>, y: &Vec<Vec<Num>>) -> Vec<Vec<Num>>{
  x.iter().zip(y.iter())
  .map(|t| sub(&t.0.to_vec(), &t.1.to_vec()))
  .collect()
}

pub fn emult(x: &Vec<Num>, y: &Vec<Num>) -> Vec<Num> {
  x.iter().zip(y.iter())
  .map(|t| t.0 * t.1)
  .collect()
}

pub fn emult2d(x: &Vec<Vec<Num>>, y: &Vec<Vec<Num>>) -> Vec<Vec<Num>>{
  x.iter().zip(y.iter())
  .map(|t| emult(&t.0.to_vec(), &t.1.to_vec()))
  .collect()
}

pub fn transpose(x: &Vec<Vec<Num>>) -> Vec<Vec<Num>> {
  let row :usize = x.len();
  let col :usize = x[0].len();
  
  (0..col)
  .map(|c| {
    (0..row)
    .map( |r| x[r][c] )
    .collect()
  })
  .collect()
}

pub fn mult (x: &Vec<Vec<Num>>, y: &Vec<Vec<Num>>) -> Vec<Vec<Num>> {
  let ty = transpose(y);

  x.iter()
  .map(|i| {
    ty.iter()
    .map(|j| dot_product(&i.to_vec(), &j.to_vec()))
    .collect()
  })
  .collect()
}

fn nif_fit<'a>(env: Env<'a>, args: &[Term<'a>])-> NifResult<Term<'a>> {
  let pid = env.pid();
  let mut my_env = OwnedEnv::new();

  let saved_list = my_env.run(|env| -> NifResult<SavedTerm> {
    let _x = args[0].in_env(env);
    let _y = args[1].in_env(env);
    let theta = args[2].in_env(env);
    let alpha = args[3].in_env(env);
    let iterations = args[4].in_env(env);
    Ok(my_env.save(make_tuple(env, &[_x, _y, theta, alpha, iterations])))
  })?;

  std::thread::spawn(move ||  {
    my_env.send_and_clear(&pid, |env| {
      let result: NifResult<Term> = (|| {
        let tuple = saved_list
        .load(env).decode::<(
          Term,
          Term,
          Term,
          Num,
          i64
        )>()?; 
        
        let x: Vec<Vec<Num>> = try!(tuple.0.decode());
        let y: Vec<Vec<Num>> = try!(tuple.1.decode());
        let theta: Vec<Vec<Num>> = try!(tuple.2.decode());
        let alpha: Num = tuple.3;
        let iterations: i64 = tuple.4;

        let tx = transpose(&x);
        let m = y.len() as Num;
        let (row, col) = (theta.len(), theta[0].len());
        let tmp = alpha/m;
        let a :Vec<Vec<Num>> = vec![vec![tmp; col]; row]; 

        let ans = (0..iterations)
          .fold( theta, |theta, _iteration|{
           sub2d(&theta, &emult2d(&mult( &tx, &sub2d( &mult( &x, &theta ), &y ) ), &a))
          });

        Ok(ans.encode(env))
      })();
      match result {
          Err(_err) => env.error_tuple("test failed".encode(env)),
          Ok(term) => term
      }  
    });
  });
  Ok(atoms::ok().to_term(env))
}

fn nif_dot_product<'a>(env: Env<'a>, args: &[Term<'a>])-> NifResult<Term<'a>> {
  let x: Vec<Num> = args[0].decode()?;
  let y: Vec<Num> = args[1].decode()?;

  Ok(dot_product(&x, &y).encode(env))
}

fn nif_sub<'a>(env: Env<'a>, args: &[Term<'a>]) -> NifResult<Term<'a>> {
  let x: Vec<Num> = args[0].decode()?;
  let y: Vec<Num> = args[1].decode()?;
  
  Ok(sub(&x, &y).encode(env))
}

fn nif_emult<'a>(env: Env<'a>, args: &[Term<'a>]) -> NifResult<Term<'a>> {
  let x: Vec<Num> = args[0].decode()?;
  let y: Vec<Num> = args[1].decode()?;
  
  Ok(emult(&x, &y).encode(env))
}

fn new(first: i64, end: i64) -> Vec<i64> {
  (first..end).collect()
}

fn nif_new<'a>(env: Env<'a>, args: &[Term<'a>]) -> NifResult<Term<'a>> {
  let first: i64 = try!(args[0].decode());
  let end: i64 = try!(args[1].decode());

  Ok(new(first, end).encode(env))
}

fn zeros<'a>(env: Env<'a>, args: &[Term<'a>]) -> NifResult<Term<'a>> {
  let len: usize = (args[0]).decode()?;
  let zero_vec = vec![0; len];
  Ok(zero_vec.encode(env))
}

