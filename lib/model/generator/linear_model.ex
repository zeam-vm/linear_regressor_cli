defmodule LinearModel do
  require Benchmark

  @doc"""

  """
  # this length of model is same as boston 
  def generator(nLabel \\ 13, nData \\ 506) do
    x = List.duplicate(0, nLabel)
    |> Enum.map(
      fn _i -> 
        LinearRegressorNif.SingleCore.new(0, nData)
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
    # theta = List.duplicate([0.0], nLabel)

    {x_train, y_train, alpha, iterations}
  end

  def rust_fit({x_train, y_train, alpha, iterations}) do
    LinearRegressorNif.SingleCore.fit( 
      x_train |> Matrix.transpose, 
      y_train |> Matrix.transpose, 
      alpha, 
      iterations )
  end

  def rayon_fit({x_train, y_train, alpha, iterations}) do
    LinearRegressorNif.MultiCore.fit( 
      x_train |> Matrix.transpose, 
      y_train |> Matrix.transpose, 
      alpha, 
      iterations )
  end

  # def to_int(num) when is_float(num) do
  #   num |> Float.floor |> Float.to_string |> Integer.parse |> elem(0)
  # end

  def variable_thread_benchmark({x_train, y_train, alpha, iterations}) do
    LinearRegressorNif.benchmark( 
      x_train |> Matrix.transpose, 
      y_train |> Matrix.transpose, 
      alpha, 
      iterations )
    receive do
      l -> l
    end
  end

  def rust_regressor(nLabel \\ 13, nData \\ 506) do
    {x_train, y_train, alpha, iterations} = setup(nLabel, nData)
    
    LinearRegressorNif.SingleCore.fit( 
      x_train |> Matrix.transpose, 
      y_train |> Matrix.transpose, 
      alpha, 
      iterations )
    |> Benchmark.time(true)
  end

  def rayon_regressor(nLabel \\ 13, nData \\ 506) do
    {x_train, y_train, alpha, iterations} = setup(nLabel, nData)

    LinearRegressorNif.MultiCore.fit( 
      x_train |> Matrix.transpose, 
      y_train |> Matrix.transpose, 
      alpha, 
      iterations )
    |> Benchmark.time(true)
  end

  def all_benchmark(nNum \\ 1, nLabel \\ 50, base \\ 1000, offset \\ 100) do
    require Integer

    num = 2*nNum
    nDatas = LinearRegressorNif.SingleCore.new(1, num)
    |> Enum.filter(& Integer.is_odd(&1))
    |> Enum.map(& &1*offset + base)

    rust_result = nDatas
    |> Enum.map(& { "50, #{&1}", rust_regressor(50, &1) |> elem(0)})

    rayon_result = nDatas
    |> Enum.map(& { "50, #{&1}", rayon_regressor(50, &1) |> elem(0)})

    # rayon_result
    ratio = 0..(length(nDatas)-1)
      |> Enum.map(& {
        "50, #{Enum.at(nDatas, &1)}",
        (Enum.at(rust_result, &1) |> elem(1))
        / (Enum.at(rayon_result, &1) |> elem(1))
        })

    ratio |> Enum.map( & { &1 |> elem(0) |> IO.inspect } )
    ratio |> Enum.map( & { &1 |> elem(1) |> IO.inspect } )
  end

  def benchmark do
    setup() |> variable_thread_benchmark
  end

end
