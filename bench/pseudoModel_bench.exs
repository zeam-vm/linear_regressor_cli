defmodule PseudoModelBench do
  use Benchfella

  # bench "Elixir", [data: setup_ex(), theta: List.duplicate([0.0], 10)] do
  #   [x_train, y_train, alpha, iterations] = data
  #   LinearRegressor.fit(x_train, y_train, theta, alpha, iterations)
  #   # |> IO.inspect
  # end

  # bench "Elixir.Inlining", [data: setup_ex(), theta: List.duplicate([0.0], 10)] do
  #   [x_train, y_train, alpha, iterations] = data
  #   LinearRegressor.Inlining.fit(x_train, y_train, theta, alpha, iterations)
  #   # |> IO.inspect
  # end

  # bench "Rust", [data: setup()] do
  #   {x_train, y_train, alpha, iterations} = data

  #   LinearRegressorNif.SingleCore.fit( 
  #     x_train , 
  #     y_train , 
  #     alpha, 
  #     iterations )
  #   # |> IO.inspect
  # end

  # bench "Optimized Rayon", [data: setup()] do
  #   {x_train, y_train, alpha, iterations} = data

  #   LinearRegressorNif.MultiCore.fit( 
  #     x_train , 
  #     y_train , 
  #     alpha, 
  #     iterations ) 
  #   # |> IO.inspect
  # end

  # bench "Filled_Rayon", [data: setup()] do
  #   [x_train, y_train, alpha, iterations] = data
  #   LinearRegressorNif.MultiCore.fit_filled_rayon(x_train, y_train, alpha, iterations)
  # end

  # defp setup do
  #   LinearModel.setup 10, 1000
  # end

  # defp setup_ex do
  # {x_train, y_train, alpha, iterations} = setup()
  # [
  #   x_train |> Matrix.transpose, 
  #   y_train |> Matrix.transpose, 
  #   alpha, 
  #   iterations
  # ]
  # end

  bench "Speed up efficiency", [] do
    LinearModel.all_benchmark 5
  end
end
