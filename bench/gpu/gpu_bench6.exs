defmodule GPUBench6 do
  use Benchfella

  bench "[Elixr)] inlining 1.024e6", [data: List.duplicate(1.0, 1024000)] do
    LinearRegressor.Inlining.dot_product(data, data)
  end  

  bench "[Rust)] single_core 1.024e6", [data: List.duplicate(1.0, 1024000)] do
    LinearRegressorNif.SingleCore.dot_product(data, data)
  end

  bench "[Rust)] multi_core 1.024e6", [data: List.duplicate(1.0, 1024000)] do
    LinearRegressorNif.MultiCore.dot_product(data, data)
  end

  bench "[Rust)] GPU 1.024e6", [data: List.duplicate(1.0, 1024000)] do
    LinearRegressorNif.GPU.dot_product(data, data)
  end
end