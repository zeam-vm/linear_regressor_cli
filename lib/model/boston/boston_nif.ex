defmodule BostonNif do

  require Benchmark

  def setup do
    features = Dataset.load_datas( "data/boston_house_prices_x.csv" )
    targets  = Dataset.load_datas( "data/boston_house_prices_y.csv" )

    x_train = 
    [ 
      features[ :crim ], 
      features[ :zn ], 
      features[ :indus ], 
      features[ :chas ], 
      features[ :nox ], 
      features[ :rm ], 
      features[ :age ], 
      features[ :dis ], 
      features[ :rad ], 
      features[ :tax ], 
      features[ :ptratio ], 
      features[ :b ], 
      features[ :lstat ], 
    ]
    |> Matrix.transpose
    y_train = [ targets[ :medv ] ] |> Matrix.transpose

    alpha = 0.0000003
    iterations = 10000
    theta = List.duplicate([0.0], length(hd x_train))

    {x_train, y_train, alpha, iterations, theta}
  end

  def rust_regressor do
    IO.puts "set up"
    {x_train, y_train, alpha, iterations, theta} = Benchmark.time setup()

    IO.puts "main process"
    theta = Benchmark.time LinearRegressorNif.fit( x_train, y_train, theta, alpha, iterations )

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
    {x_train, y_train, alpha, iterations, theta} = Benchmark.time setup()

    IO.puts "main process"
    theta = Benchmark.time LinearRegressorNif.rayon_fit( x_train, y_train, theta, alpha, iterations )

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
