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
    ("_norum", 1, nif_norum),
    ("_zeros", 1, zeros),
    ("_new", 2, nif_new),
    ("_sub", 2, nif_sub),
    ("_emult", 2, nif_emult),
    ("_fit", 5, nif_fit), 
    ("gpuinfo", 0, gpuinfo),
    ("_call_ocl_dp", 2, call_ocl_dp),
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

pub fn norum(x: &Vec<f64>) -> f64 {
  x.iter()
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

pub fn call_ocl_dp<'a>(env: Env<'a>, args: &[Term<'a>]) -> NifResult<Term<'a>> {
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
  use ocl::{Platform, Device, Context, Queue, Program,Buffer, Kernel};
  use ocl::enums::{DeviceInfo, DeviceInfoResult, KernelWorkGroupInfo, KernelWorkGroupInfoResult};

  let src = r#"
    __kernel void dot_product(
      __global double* x, 
      __global double* y,
      __global double* partial_sums
    ){
      const int LOCAL_ID = get_local_id(0);
      const int GROUP_SIZE = get_local_size(0);
      const int GROUP_ID = get_group_id(0);
      const size_t PRIVATE_SIZE = 4;

      // Copy to private memory
      double tmp[4];

      for(size_t i = 0; i < PRIVATE_SIZE; ++i){
      	tmp[i] = x[GROUP_ID + LOCAL_ID + i] * y[GROUP_ID + LOCAL_ID + i];
      }	

      // Sync
      barrier(CLK_LOCAL_MEM_FENCE);
      
      // Calculate inner product
      double tmp1 = 0.0;
      for(size_t i = 0; i < PRIVATE_SIZE; ++i){
      	tmp1 += tmp[i];
      }

      partial_sums[ GROUP_ID ] = tmp1;
    }
  "#;

  let vec_size = x.len();
  let f64_size = 8;
  let platform = Platform::default(); //OSの情報とか
  let device = Device::by_idx_wrap(platform, 0).unwrap();

  println!("Device Name:{:?}", device.name());
  println!("Device Vendor:{:?}", device.vendor());

  let context = Context::builder()
    .platform(platform)
    .devices(device.clone())
    .build()?;
  let program = Program::builder()
    .devices(device)
    .src(src)
    .build(&context).unwrap();
  let queue = Queue::new(&context, device, None)?;

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
  println!("local_memory_size:{:?}", local_memory_size);
  println!("local_array_size:{:?}", local_array_size);

  let source_buffer_x = Buffer::builder()
    .queue(queue.clone())
    .flags(MemFlags::new().read_only())
    .len(vec_size)
    .copy_host_slice(&x)
    .build()?;

  let source_buffer_y = Buffer::builder()
		.queue(queue.clone())
    .flags(MemFlags::new().read_only())
    .len(vec_size)
    .copy_host_slice(&y)
    .build()?;

  let work_group_size = 1;
  let private_size = 4;
  let work_group_num = vec_size / private_size;
  println!("work_group_size:{:?}", work_group_size);
  println!("work_group_num:{:?}", work_group_num);

  let output_buffer = Buffer::<f64>::builder()
    .queue(queue.clone())
    .flags(MemFlags::new().write_only())
    .len(work_group_num)
    .build()?;

  let kernel : ocl::Kernel = Kernel::builder()
    .program(&program)
    .name("dot_product")
    .queue(queue.clone())
    .arg(&source_buffer_x) 
    .arg(&source_buffer_y)
    .arg(&output_buffer)
    // .arg(&private_size)
    .build().unwrap();


  let preferred_work_group_size_multiple: usize = match kernel.wg_info(device, KernelWorkGroupInfo::PreferredWorkGroupSizeMultiple){
    Ok(KernelWorkGroupInfoResult::PreferredWorkGroupSizeMultiple(res)) => res,
    _ => { 
      println!("failed to get DeviceInfoResult::PreferredWorkGroupSizeMultiple");
      1
    },
  };

  println!("PreferredWorkGroupSizeMultiple:{:?}", preferred_work_group_size_multiple);

  let res: ocl::Result<()>;
  unsafe { 
     res = kernel.cmd()
      .queue(&queue)
      .global_work_offset(kernel.default_global_work_offset())
      .global_work_size(work_group_num)
      .local_work_size(work_group_size)
      .enq();
  }

  match res {
    Ok(_) => {
      println!("success execute kernel code");
      let mut result = vec![0f64; work_group_num];
      output_buffer.cmd()
      	.queue(&queue)
      	.offset(0)
      	.read(&mut result)
      	.enq()?;
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
        
        let x: Vec<f64> = try!(tuple.0.decode());
        let y: Vec<f64> = try!(tuple.1.decode());

        Ok(dot_product(&x, &y).encode(env))
      })();
      match result {
          Err(_err) => env.error_tuple("test failed".encode(env)),
          Ok(term) => term
      }  
    });
  });
  Ok(atoms::ok().to_term(env))
}

fn nif_norum<'a>(env: Env<'a>, args: &[Term<'a>])-> NifResult<Term<'a>> {
  // let x :Vec<f64> = try!(args[0].decode());
  // Ok(norum(&x).encode(env))

  // println!("{:?}", x);

  let pid = env.pid();
  let mut my_env = OwnedEnv::new();
  let saved_list = my_env.run(|env| -> NifResult<SavedTerm> {
    let _x = args[0].in_env(env);
    let _x1 = args[0].in_env(env);
    Ok(my_env.save(make_tuple(env, &[_x, _x1])))
  })?;

  std::thread::spawn(move ||  {
    my_env.send_and_clear(&pid, |env| {
      let result: NifResult<Term> = (|| {
        let tuple = saved_list
        .load(env).decode::<(
          Term,
          Term
        )>()?; 
        
        let x: Vec<f64> = try!(tuple.0.decode());

        Ok(norum(&x).encode(env))
      })();
      match result {
          Err(_err) => env.error_tuple("test failed".encode(env)),
          Ok(term) => term
      }  
    });
  });
  Ok(atoms::ok().to_term(env))
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

