defmodule BostonNif do
  require Benchmark

  def rust_regressor do
    IO.puts "set up"
    [x_train, y_train, alpha, iterations] = Benchmark.time Boston.setup()

    IO.puts "main process"
    theta = Benchmark.time LinearRegressorNif.SingleCore.fit( x_train, y_train, alpha, iterations )

    IO.puts "theta"
    IO.inspect theta
    
    x_test = [ [ 0.00632 ], [ 18.0 ], [ 2.31 ], [ 0.0 ], [ 0.538 ], [ 6.575 ], [ 65.2 ], [ 4.09 ], [ 1.0 ], [ 296.0 ], [ 15.3 ], [ 396.9 ], [ 4.98 ] ] |> Matrix.transpose
    y_test = [ [ 24.0 ] ]

    predicted = LinearRegressor.predict( x_test, theta )

    error = LinearRegressor.cost( x_test, y_test, theta )

    IO.puts "y_test:  #{ y_test    |> inspect }"
    IO.puts "predict: #{ predicted |> inspect }"
    IO.puts ""
    IO.puts "error:"
    IO.inspect error
  end

  def rayon_regressor do
    IO.puts "set up"
    {x_train, y_train, alpha, iterations} = Benchmark.time Boston.setup()

    IO.puts "main process"
    theta = Benchmark.time LinearRegressorNif.MultiCore.fit( x_train, y_train, alpha, iterations )

    IO.puts "theta"
    IO.inspect theta
    
    x_test = [ [ 0.00632 ], [ 18.0 ], [ 2.31 ], [ 0.0 ], [ 0.538 ], [ 6.575 ], [ 65.2 ], [ 4.09 ], [ 1.0 ], [ 296.0 ], [ 15.3 ], [ 396.9 ], [ 4.98 ] ] |> Matrix.transpose
    y_test = [ [ 24.0 ] ]

    predicted = LinearRegressor.predict( x_test, theta )

    error = LinearRegressor.cost( x_test, y_test, theta )

    IO.puts "y_test:  #{ y_test    |> inspect }"
    IO.puts "predict: #{ predicted |> inspect }"
    IO.puts ""
    IO.puts "error:"
    IO.inspect error
  end
end
