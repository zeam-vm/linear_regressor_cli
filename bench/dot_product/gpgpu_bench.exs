defmodule GPUBench do
  use Benchfella

  bench "[Rust] GPU 1.024e3", [data: List.duplicate(1.0, 1024)] do
    LinearRegressorNif.GPU.dot_product(data, data)
  end

  bench "[Rust] GPU 1.024e4", [data: List.duplicate(1.0, 10240)] do
    LinearRegressorNif.GPU.dot_product(data, data)
  end

  bench "[Rust] GPU 1.024e5", [data: List.duplicate(1.0, 102400)] do
    LinearRegressorNif.GPU.dot_product(data, data)
  end

  bench "[Rust] GPU 1.024e6", [data: List.duplicate(1.0, 1024000)] do
    LinearRegressorNif.GPU.dot_product(data, data)
  end

  bench "[Rust] GPU 1.024e7", [data: List.duplicate(1.0, 10240000)] do
    LinearRegressorNif.GPU.dot_product(data, data)
  end
end
