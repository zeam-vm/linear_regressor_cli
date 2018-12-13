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

  def rust_regressor(nLabel \\ 13, nData \\ 506) do
    IO.puts "set up"
    {x_train, y_train, alpha, iterations, theta} = setup(nLabel, nData)
    |> Benchmark.time 

    IO.puts "main process"
    LinearRegressorNif.fit( 
      x_train |> Matrix.transpose, 
      y_train |> Matrix.transpose, 
      theta, 
      alpha, 
      iterations )
    |> Benchmark.time(true)

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
    LinearRegressorNif.rayon_fit( 
      x_train |> Matrix.transpose, 
      y_train |> Matrix.transpose, 
      theta, 
      alpha, 
      iterations )
    |> Benchmark.time(true)

    # IO.puts "theta"
    # IO.inspect theta
  end

  def inline_regressor(nLabel \\ 13, nData \\ 506) do
    IO.puts "set up"
    {x_train, y_train, alpha, iterations, theta} = setup(nLabel, nData)
    |> Benchmark.time 

    IO.puts "main process"
    LinearRegressorInlining.rayon_fit( 
      x_train |> Matrix.transpose, 
      y_train |> Matrix.transpose, 
      theta, 
      alpha, 
      iterations )
    |> Benchmark.time(true)

    # IO.puts "theta"
    # IO.inspect theta
    end

    def all_benchmark(nNum \\ 1, nLabel \\ 50, offset \\ 100) do
      require Integer

      num = 2*nNum
      nDatas = LinearRegressorNif._new(1, num)
      |> Enum.filter(& Integer.is_odd(&1))
      |> Enum.map(& &1*offset)

      rust_result = nDatas
      |> Enum.map(& { "50, #{&1}", rust_regressor(50, &1) |> elem(0)})

      rayon_result = nDatas
      |> Enum.map(& { "50, #{&1}", rayon_regressor(50, &1) |> elem(0)})

      rayon_result
      ratio = 0..(length(nDatas)-1)
        |> Enum.map(& {
          "50, #{Enum.at(nDatas, &1)}",
          (Enum.at(rust_result, &1) |> elem(1))
          / (Enum.at(rayon_result, &1) |> elem(1))
          })

      ratio
    end

    def to_int(num) when is_float(num) do
      num |> Float.floor |> Float.to_string |> Integer.parse |> elem(0)
    end
end
