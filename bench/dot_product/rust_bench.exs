defmodule RustBench do
  use Benchfella

  bench "[Rust] single_core 1.024e3", [data: List.duplicate(1.0, 1024)] do
    LinearRegressorNif.SingleCore.dot_product(data, data)
  end

  bench "[Rust] single_core 1.024e4", [data: List.duplicate(1.0, 10240)] do
    LinearRegressorNif.SingleCore.dot_product(data, data)
  end

  bench "[Rust] single_core 1.024e5", [data: List.duplicate(1.0, 102400)] do
    LinearRegressorNif.SingleCore.dot_product(data, data)
  end

  bench "[Rust] single_core 1.024e6", [data: List.duplicate(1.0, 1024000)] do
    LinearRegressorNif.SingleCore.dot_product(data, data)
  end

  bench "[Rust] single_core 1.024e7", [data: List.duplicate(1.0, 10240000)] do
    LinearRegressorNif.SingleCore.dot_product(data, data)
  end
end