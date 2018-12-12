defmodule LinearModel do
  require Benchmark

  @doc"""

  """

  # y = x + noize
  # 0 <= x < 100
  # this length of model is same as boston 
  def generator(nLabel \\ 13, nData \\ 506) do
    x = List.duplicate(0, nLabel)
    |> Enum.map(
      fn _i -> 
        LinearRegressorNif._new(0, nData)
        |> Enum.map(& &1*:rand.uniform)
        |> Enum.sort
      end)

    sum = x |> Matrix.transpose 
    |> Enum.map(& &1 |> Enum.sum |> Kernel./(nLabel))

    y = sum 
    |> Enum.map(& &1 + :rand.normal)
    |> Enum.sort

    {x, [y]}
  end

  def setup(nLabel \\ 13, nData \\ 506) do
    {x_train, y_train} = generator(nLabel, nData)

    alpha = 0.0000003
    iterations = 10000
    theta = List.duplicate([0.0], nLabel)

    {x_train, y_train, alpha, iterations, theta}
  end

  def predict do

  end

  def nif_regressor(nLabel \\ 13, nData \\ 506) do
    IO.puts "set up"
    {x_train, y_train, alpha, iterations, theta} = setup(nLabel, nData)
    |> Benchmark.time 

    IO.puts "main process"
    theta = Benchmark.time LinearRegressorNif.fit( 
      x_train |> Matrix.transpose, 
      y_train |> Matrix.transpose, 
      theta, 
      alpha, 
      iterations )

    # IO.puts "theta"
    # IO.inspect theta
    
    # x_test = List.duplicate([1.0], nLabel) 
    # |> Matrix.transpose
    # y_test = [ [ 1.0 ] ]

    # predicted = LinearRegressor.predict( x_test, theta )

    # error = LinearRegressor.cost( x_test, y_test, theta )

    # IO.puts "y_test:  #{ y_test    |> inspect }"
    # IO.puts "predict: #{ predicted |> inspect }"
    # IO.puts ""
    # IO.puts "error: "
    # IO.inspect error
  end

  def rayon_regressor(nLabel \\ 13, nData \\ 506) do
    IO.puts "set up"
    {x_train, y_train, alpha, iterations, theta} = setup(nLabel, nData)
    |> Benchmark.time 

    IO.puts "main process"
    theta = Benchmark.time LinearRegressorNif.rayon_fit( 
      x_train |> Matrix.transpose, 
      y_train |> Matrix.transpose, 
      theta, 
      alpha, 
      iterations )

    # IO.puts "theta"
    # IO.inspect theta
  end

  def inline_regressor(nLabel \\ 13, nData \\ 506) do
    IO.puts "set up"
    {x_train, y_train, alpha, iterations, theta} = setup(nLabel, nData)
    |> Benchmark.time 

    IO.puts "main process"
    theta = Benchmark.time LinearRegressorInlining.fit( 
      x_train |> Matrix.transpose, 
      y_train |> Matrix.transpose, 
      theta, 
      alpha, 
      iterations )

    # IO.puts "theta"
    # IO.inspect theta
    end
end
