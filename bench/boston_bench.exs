defmodule BostonModelBench do
  use Benchfella

  # bench "Elixir", [data: setup_ex(), theta: List.duplicate([0.0], 13)] do
  #   [x_train, y_train, alpha, iterations] = data
  #   LinearRegressor.fit(x_train, y_train, theta, alpha, iterations)
  # end

  # bench "Elixir_inlining", [data: setup_ex(), theta: List.duplicate([0.0], 13)] do
  #   [x_train, y_train, alpha, iterations] = data
  #   LinearRegressor.Inlining.fit(x_train, y_train, theta, alpha, iterations)
  # end

  # bench "Rustler_single_core", [data: setup()] do
  #   [x_train, y_train, alpha, iterations] = data
  #   LinearRegressorNif.SingleCore.fit(x_train, y_train, alpha, iterations)
  # end

  # bench "Rustler_mc_optimized", [data: setup()] do
  #   [x_train, y_train, alpha, iterations] = data
  #   LinearRegressorNif.MultiCore.fit(x_train, y_train, alpha, iterations)
  # end
  
  # bench "Rustler_multi_core", [data: setup()] do
  #   [x_train, y_train, alpha, iterations] = data
  #   LinearRegressorNif.MultiCore.fit_filled_rayon(x_train, y_train, alpha, iterations)
  # end

  # defp setup do
  #   Boston.setup
  # end

  # defp setup_ex do
  # [x_train, y_train, alpha, iterations] = Boston.setup
  # [
  #   x_train |> Matrix.transpose, 
  #   y_train |> Matrix.transpose, 
  #   alpha, 
  #   iterations
  # ]
  # end
end