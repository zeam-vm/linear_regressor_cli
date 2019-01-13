defmodule LinearModel do
  require Benchmark

  @nlabel 10
  @ndata 1000

  @doc"""

  """
  def o_n(x, y) do
    (4*x*y + y) / x
  end

  def efficiency(x, y) do
    ratio = o_n(x, y)
    1 / (1 - 0.01*ratio)
  end

  # this length of model is same as boston 
  def generator(nLabel \\ @nlabel, nData \\ @ndata) do
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

  def setup(nLabel \\ @nlabel, nData \\ @ndata) do
    {x_train, y_train} = generator(nLabel, nData)

    alpha = 0.0000003
    iterations = 10000
    # theta = List.duplicate([0.0], nLabel)

    {x_train, y_train, alpha, iterations}
  end

  def rust_regressor(nLabel \\ @nlabel, nData \\ @ndata) do
    {x_train, y_train, alpha, iterations} = setup(nLabel, nData)
    
    LinearRegressorNif.SingleCore.fit( 
      x_train , 
      y_train , 
      alpha, 
      iterations )
    |> Benchmark.time(true)
  end

  def rayon_regressor(nLabel \\ @nlabel, nData \\ @ndata) do
    {x_train, y_train, alpha, iterations} = setup(nLabel, nData)

    LinearRegressorNif.MultiCore.fit( 
      x_train , 
      y_train , 
      alpha, 
      iterations )
    |> Benchmark.time(true)
  end

  def all_benchmark(nNum \\ 1, base \\ @ndata, offset \\ 500, nLabel \\ @nlabel) do
    require Integer

    IO.puts "predict efficiency:#{efficiency(nLabel, base)}"

    nDatas = LinearRegressorNif.SingleCore.new(0, nNum)
    |> Enum.map(& &1*offset + base)

    rust_result = nDatas
    |> Enum.map(& { "#{nLabel}, #{&1}", rust_regressor(nLabel, &1) |> elem(0)})

    rayon_result = nDatas
    |> Enum.map(& { "#{nLabel}, #{&1}", rayon_regressor(nLabel, &1) |> elem(0)})

    # rayon_result
    ratio = 0..(nNum-1)
      |> Enum.map(& {
        "#{nLabel} #{Enum.at(nDatas, &1)}",
        (Enum.at(rust_result, &1) |> elem(1))
        / (Enum.at(rayon_result, &1) |> elem(1))
        })

    ratio |> Enum.map( & &1 |> elem(0) |> IO.inspect )
    ratio |> Enum.map( & &1 |> elem(1) |> IO.inspect )
  end

  def rust_benchmark(nNum \\ 1, base \\ @ndata, offset \\ 500, nLabel \\ @nlabel) do
    LinearRegressorNif.SingleCore.new(0, nNum)
    |> Enum.map(& &1*offset + base)
    |> Enum.map(& { "#{nLabel}, #{&1}", rust_regressor(nLabel, &1) |> elem(0)})
  end

  def rayon_benchmark(nNum \\ 1, base \\ @ndata, offset \\ 500, nLabel \\ @nlabel) do
    LinearRegressorNif.SingleCore.new(0, nNum)
    |> Enum.map(& &1*offset + base)
    |> Enum.map(& { "#{nLabel}, #{&1}", rayon_regressor(nLabel, &1) |> elem(0)})
  end
end
