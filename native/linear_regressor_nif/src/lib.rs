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

lazy_static! {
    static ref POOL:scoped_pool::Pool = scoped_pool::Pool::new(2);
}

pub fn dot_product(x: &Vec<f64>, y: &Vec<f64>) -> f64 {
  x.iter().zip(y.iter())
  .map(|t| t.0 * t.1)
  .fold(0.0, |sum, i| sum + i)
}

pub fn gpuinfo<'a>(env: Env<'a>, _args: &[Term<'a>]) -> NifResult<Term<'a>> {
  use ocl::enums::{PlatformInfo};
  use ocl::{Platform, Device};

  let platforms = Platform::list();

  println!("platform num:{:?}", platforms.len());

  for p_idx in 0..platforms.len() {
  	let platform = &platforms[p_idx];
  	let devices = Device::list_all(platform).unwrap();

	if devices.is_empty() { continue; }	

	println!("Profile:{:?}", platform.info(PlatformInfo::Profile));  
	println!("Version:{:?}", platform.info(PlatformInfo::Version));
	println!("Name:{:?}", platform.info(PlatformInfo::Name));
	println!("Vendor:{:?}", platform.info(PlatformInfo::Vendor));
	
	for device in devices.iter() {
		  println!("Device Name:{:?}", device.name());
			println!("Device Vendor:{:?}", device.vendor());	
    }
  }

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

  POOL.spawn(move ||  {
    my_env.send_and_clear(&pid, |env| {
      let result: NifResult<Term> = (|| {
        let tuple = saved_list
        .load(env).decode::<(
          Term,
          Term
        )>()?; 
        
        let x: Vec<f64> = try!(tuple.0.decode());
        let y: Vec<f64> = try!(tuple.1.decode());

        let result: ocl::Result<(Vec<f64>)> = dot_product_ocl(&x, &y);
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

pub fn dot_product_ocl(x: &Vec<f64>, y: &Vec<f64>) -> ocl::Result<(Vec<f64>)> {
  use ocl::{Platform, Device};
  use ocl::enums::{DeviceInfo, DeviceInfoResult};

  let vec_size = x.len();
  let platform = Platform::default();
  let device = Device::first(platform).unwrap(); 
  let f64_size :usize = 8;
  let kernel_vector_dim :usize = 1;

  // 並列化の最大次元数 1 -> ベクトル
  let work_dim : u32 = match device.info(DeviceInfo::MaxWorkItemDimensions){
    Ok(DeviceInfoResult::MaxWorkItemDimensions(res)) => res,
    _ => { 
      println!("failed to get DeviceInfoResult::MaxWorkItemDimensions");
      1
    },
  };

  //ワークグループ総数
  let max_work_group_size: usize = match device.info(DeviceInfo::MaxWorkGroupSize){
    Ok(DeviceInfoResult::MaxWorkGroupSize(res)) => res,
    _ => { 
      println!("failed to get DeviceInfoResult::MaxWorkGroupSize");
      1
    },
  };

  // f64 of Compute Unit(演算ユニット数) ＝ Streaming Multiprocessor(SM) 
  let compute_unit_num :u32 = match device.info(DeviceInfo::MaxComputeUnits){
    Ok(DeviceInfoResult::MaxComputeUnits(res)) => res,
    _ => { 
      println!("failed to get DeviceInfoResult::MaxComputeUnits");
      0
    },
  };

  // ローカルメモリ(スクラッチパッドメモリ)の最大容量
  // このサイズを超えたデータをローカルメモリ(__local修飾子)としてカーネルに渡すと
  // エラーが発生する．
  let max_local_memory_size :u32 = match device.info(DeviceInfo::LocalMemSize){
    Ok(DeviceInfoResult::LocalMemSize(res)) => res as u32,
    _ => { 
      println!("failed to get DeviceInfoResult::LocalMemSize");
      0
    },
  };

  // ワークグループあたりのワークアイテム数？
  let max_work_item_size: Vec<usize> = match device.info(DeviceInfo::MaxWorkItemSizes){
    Ok(DeviceInfoResult::MaxWorkItemSizes(res)) => res,
    _ => { 
      println!("failed to get DeviceInfoResult::MaxWorkGroupSize");
      vec![1; 3]
    },
  };

  // ローカルワークアイテムサイズ
  // 1つのワークグループにつき処理されるワーク数の最適値
  let work_item_num = (vec_size as u32 + compute_unit_num-1) / compute_unit_num;
  
  // let global_work_size = vec_size / kernel_vector_dim;
  let global_work_size = (work_item_num * compute_unit_num) as usize;
  let local_work_size = work_item_num as usize;

  assert_eq!(global_work_size % local_work_size, 0);

  // ローカルメモリーサイズ
  let local_memory_size = match max_local_memory_size > (vec_size * f64_size) as u32{
    true => vec_size*f64_size,
    false => max_local_memory_size as usize,
  }; 

  let local_array_size  = (local_memory_size + f64_size - 1)/ f64_size;

  println!("input_size:{:?}", vec_size);

  println!("work_dim:	{:?}", work_dim);
  println!("compute_unit_num:	{:?}", compute_unit_num);
  println!("max_work_group_size:{:?}", max_work_group_size);
  println!("max_local_memory_size:{:?}", max_local_memory_size);
  println!("max_work_item_size:{:?}", max_work_item_size);
  println!("work_item_num:{:?}", work_item_num);
  println!("global_work_size:{:?}", global_work_size);
  println!("local_memory_size:{:?}", local_memory_size);
  println!("local_array_size:{:?}", local_array_size);
  
  println!("local_work_size:{:?}", local_work_size);
  println!("global_work_size:{:?}", global_work_size);

  let src = r#"
    __kernel void dot_product4(
      __global double4* x, 
      __global double4* y,
      __global double* output,
      int array_size,
      __local double4* partial_sums
    ){
      int lid = get_local_id(0);
      int gid = get_global_id(0);
      int offset = get_local_size(0);

      // printf("group_size:%d\n", offset);
      // printf("x*y[%d] = %lf\n", gid, x[gid]*y[gid]);

      partial_sums[lid] = x[gid] * y[gid];

      // printf("partial_sums[%d]= %lf\n", lid, partial_sums[lid]);

      for(int i = offset >> 1; i > 0; i >>= 1) {
          barrier(CLK_LOCAL_MEM_FENCE);

          if(lid < i) {
              partial_sums[lid] += partial_sums[lid + i];
          }
      }

      // printf("group_id[%d]\n", get_group_id(0));

      if(lid == 0) {
        output[get_group_id(0)] = dot(partial_sums[0], (double4)1.0);
      }
    }

    __kernel void dot_product1(
      __global double* x, 
      __global double* y,
      __global double* output,
      int array_size,
      __local double* partial_sums
    ){
      int lid = get_local_id(0);
      int gid = get_global_id(0);
      int offset = get_local_size(0);

      if(gid >= array_size){
        return;
      }

      // printf("group_size:%d\n", offset);
      // printf("x*y[%d] = %lf\n", gid, x[gid]*y[gid]);

      partial_sums[lid] = x[gid] * y[gid];

      // printf("partial_sums[%d]= %lf\n", lid, partial_sums[lid]);

      for(int i = offset >> 1; i > 0; i >>= 1) {
          barrier(CLK_LOCAL_MEM_FENCE);

          if(lid < i) {
              partial_sums[lid] += partial_sums[lid + i];
          }
      }

      // printf("group_id[%d]\n", get_group_id(0));

      if(lid == 0) {
        output[get_group_id(0)] = partial_sums[0];
      }
    }
  "#;

  let pro_que = ProQue::builder()
    .src(src)
    .dims(1)
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

  let output_buffer = Buffer::<f64>::builder()
    .queue(pro_que.queue().clone())
    .flags(MemFlags::new().write_only())
    .len(vec_size)
    .fill_val(0f64)
    .build()?;

  let kernel : ocl::Kernel = 
    pro_que.kernel_builder(
      ["dot_product", &kernel_vector_dim.to_string()].concat())
    .arg(&source_buffer_x) // immutable
    .arg(&source_buffer_y) // immutable
    .arg(&output_buffer) // immutable
    .arg(&vec_size) // immutable
    .arg_local::<f64>(local_work_size as usize)
    .build()?;

  // OpenCL Debug
  // let _res = pro_que.queue().flush();

  let res: ocl::Result<()>;
  unsafe { 
    res = kernel.cmd()
      .queue(pro_que.queue())
      .global_work_offset(kernel.default_global_work_offset())
      .global_work_size(global_work_size)
      // .local_work_size(kernel.default_local_work_size())
      .local_work_size(local_work_size)
      .enq();
  }

  println!("execute kernel.enq!()");

  match res {
    Ok(_) => {
      println!("success execute kernel code");
      let mut result = vec![0f64; vec_size];
      output_buffer.read(&mut result).enq()?;
      Ok(result)
    },
    Err(err) => {
      println!("{:?}", err);
      Err(err)
    },
  }
}

pub fn sub(x: &Vec<f64>, y: &Vec<f64>) -> Vec<f64> {
  x.iter().zip(y.iter())
  .map(|t| t.0 - t.1)
  .collect()
}

pub fn sub2d(x: &Vec<Vec<f64>>, y: &Vec<Vec<f64>>) -> Vec<Vec<f64>>{
  x.iter().zip(y.iter())
  .map(|t| sub(&t.0.to_vec(), &t.1.to_vec()))
  .collect()
}

pub fn emult(x: &Vec<f64>, y: &Vec<f64>) -> Vec<f64> {
  x.iter().zip(y.iter())
  .map(|t| t.0 * t.1)
  .collect()
}

pub fn emult2d(x: &Vec<Vec<f64>>, y: &Vec<Vec<f64>>) -> Vec<Vec<f64>>{
  x.iter().zip(y.iter())
  .map(|t| emult(&t.0.to_vec(), &t.1.to_vec()))
  .collect()
}

pub fn transpose(x: &Vec<Vec<f64>>) -> Vec<Vec<f64>> {
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

pub fn mult (x: &Vec<Vec<f64>>, y: &Vec<Vec<f64>>) -> Vec<Vec<f64>> {
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
          f64,
          i64
        )>()?; 
        
        let x: Vec<Vec<f64>> = try!(tuple.0.decode());
        let y: Vec<Vec<f64>> = try!(tuple.1.decode());
        let theta: Vec<Vec<f64>> = try!(tuple.2.decode());
        let alpha: f64 = tuple.3;
        let iterations: i64 = tuple.4;

        let tx = transpose(&x);
        let m = y.len() as f64;
        let (row, col) = (theta.len(), theta[0].len());
        let tmp = alpha/m;
        let a :Vec<Vec<f64>> = vec![vec![tmp; col]; row]; 

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
  let x: Vec<f64> = args[0].decode()?;
  let y: Vec<f64> = args[1].decode()?;

  Ok(dot_product(&x, &y).encode(env))
}

fn nif_sub<'a>(env: Env<'a>, args: &[Term<'a>]) -> NifResult<Term<'a>> {
  let x: Vec<f64> = args[0].decode()?;
  let y: Vec<f64> = args[1].decode()?;
  
  Ok(sub(&x, &y).encode(env))
}

fn nif_emult<'a>(env: Env<'a>, args: &[Term<'a>]) -> NifResult<Term<'a>> {
  let x: Vec<f64> = args[0].decode()?;
  let y: Vec<f64> = args[1].decode()?;
  
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

