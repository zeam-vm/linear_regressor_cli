defmodule ElixirBench do
  use Benchfella

  bench "[Elixr] inlining 1.024e3", [data: List.duplicate(1.0, 1024)] do
    LinearRegressor.Inlining.dot_product(data, data)
  end

  bench "[Elixr] inlining 1.024e4", [data: List.duplicate(1.0, 10240)] do
    LinearRegressor.Inlining.dot_product(data, data)
  end

  bench "[Elixr] inlining 1.024e5", [data: List.duplicate(1.0, 102400)] do
    LinearRegressor.Inlining.dot_product(data, data)
  end
  
  bench "[Elixr] inlining 1.024e6", [data: List.duplicate(1.0, 1024000)] do
    LinearRegressor.Inlining.dot_product(data, data)
  end
  
  bench "[Elixr] inlining 1.024e7", [data: List.duplicate(1.0, 10240000)] do
    LinearRegressor.Inlining.dot_product(data, data)
  end
end