defmodule BostonModelBench do
  use Benchfella

  bench "Elixir", [data: setup(), theta: List.duplicate([0.0], 13)] do
    [x_train, y_train, alpha, iterations] = data
    LinearRegressor.fit(x_train, y_train, theta, alpha, iterations)
  end

  bench "Elixir.Inlining", [data: setup(), theta: List.duplicate([0.0], 13)] do
    [x_train, y_train, alpha, iterations] = data
    LinearRegressor.Inlining.fit(x_train, y_train, theta, alpha, iterations)
  end

  bench "Rust", [data: setup()] do
    [x_train, y_train, alpha, iterations] = data
    LinearRegressorNif.SingleCore.fit(x_train, y_train, alpha, iterations)
  end

  bench "Rust.Rayon", [data: setup()] do
    [x_train, y_train, alpha, iterations] = data
    LinearRegressorNif.MultiCore.fit(x_train, y_train, alpha, iterations)
  end
  
  defp setup do
    Boston.setup
  end
end