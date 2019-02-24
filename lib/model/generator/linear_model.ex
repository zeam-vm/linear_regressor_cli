defmodule LinearModel do
  @doc """

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
end

defmodule LinearModel.Params do
  def nlabel,  do: 10
  def ndata, do: 1000
end

defmodule LinearModel.Regressor do 
  require Benchmark
  import LinearModel
  alias LinearModel.Params

  @doc """
  
  """
  def purelixir(nLabel \\ Params.nlabel, nData \\ Params.ndata) do
    
  end

  @doc """
  
  """
  def elixir_inlining(nLabel \\ Params.nlabel, nData \\ Params.ndata) do
    
  end


  @doc """
  
  """
  def rust(nLabel \\ Params.nlabel, nData \\ Params.ndata) do
    {x_train, y_train, alpha, iterations} = setup(nLabel, nData)
    
    LinearRegressorNif.SingleCore.fit( 
      x_train , 
      y_train , 
      alpha, 
      iterations )
    |> Benchmark.time(true)
  end

  @doc """
  A part of functions are filled by rayon because this is faster.
  """
  def part_rayon(nLabel \\ Params.nlabel, nData \\ Params.ndata) do
    {x_train, y_train, alpha, iterations} = setup(nLabel, nData)

    LinearRegressorNif.MultiCore.fit( 
      x_train , 
      y_train , 
      alpha, 
      iterations )
    |> Benchmark.time(true)
  end

  @doc """
  all functions are filled by rayon
  """
  def filled_rayon(nLabel \\ Params.nlabel, nData \\ Params.ndata) do
    {x_train, y_train, alpha, iterations} = setup(nLabel, nData)

    LinearRegressorNif.MultiCore.fit_filled_rayon( 
      x_train , 
      y_train , 
      alpha, 
      iterations )
    |> Benchmark.time(true)
  end
end

defmodule LinearModel.Benchmark do
  require Benchmark
  alias LinearModel.Params
  
  @moduledoc """
  benchmarks for linear regression.
  """

  @doc """
  Benchmark linear regression by rust and rayon.
  Also distplay sppedup efficiency.
  """
  def all_benchmark(nNum \\ 1, base \\ Params.ndata, offset \\ 1000, nLabel \\ Params.nlabel) do
    require Integer

    nDatas = LinearRegressorNif.SingleCore.new(0, nNum)
    |> Enum.map(& &1*offset + base)

    rust_result = nDatas
    |> Enum.map(& { "#{nLabel}, #{&1}", LinearModel.Regressor.rust(nLabel, &1) |> elem(0)})

    rayon_result = nDatas
    |> Enum.map(& { "#{nLabel}, #{&1}", LinearModel.Regressor.part_rayon(nLabel, &1) |> elem(0)})

    # filled_rayon_result = nDatas
    # |> Enum.map(& { "#{nLabel}, #{&1}", filled_rayon(nLabel, &1) |> elem(0)})

    # rayon_result
    ratio_little_rayon = 0..(nNum-1)
      |> Enum.map(& {
        "#{nLabel} #{Enum.at(nDatas, &1)}",
        (Enum.at(rust_result, &1) |> elem(1))
        / (Enum.at(rayon_result, &1) |> elem(1))
        })


    # ratio_little_rayon |> Enum.map( & &1 |> elem(0) |> IO.inspect )
    # ratio_little_rayon |> Enum.map( & &1 |> elem(1) |> IO.inspect )

    # ratio_filled_rayon = 0..(nNum-1)
    #   |> Enum.map(& {
    #     "#{nLabel} #{Enum.at(nDatas, &1)}",
    #     (Enum.at(rust_result, &1) |> elem(1))
    #     / (Enum.at(filled_rayon_result, &1) |> elem(1))
    #     })

    # ratio_filled_rayon |> Enum.map( & &1 |> elem(0) |> IO.inspect )
    # ratio_filled_rayon |> Enum.map( & &1 |> elem(1) |> IO.inspect )

    result = 0..(nNum-1)
    |> Enum.map(& 
        [
          nLabel,
          Enum.at(nDatas, &1),
          Enum.at(rust_result, &1) |> elem(1) |> Kernel./(1_000_000),
          Enum.at(rayon_result, &1) |> elem(1) |> Kernel./(1_000_000),
          # ,(Enum.at(filled_rayon_result, &1) |> elem(1)),
          Enum.at(ratio_little_rayon, &1) |> elem(1)
        ])

    "ratio.txt"|> File.write( 
      result 
      |> Enum.map(& (&1 |> Enum.map(fn x ->  "& #{x} " end)) ++ ["\n"] )
        )
  end

  @doc """
  Benchmark linear regression by rust and rayon.
  Also distplay sppedup efficiency.
  """
  def benchmarks(nNum \\ 1, base \\ Params.ndata, offset \\ 1000, nLabel \\ Params.nlabel) do
    nDatas = LinearRegressorNif.SingleCore.new(0, nNum)
    |> Enum.map(& &1*offset + base)

    rust_result = nDatas
    |> Enum.map(& { "#{nLabel}, #{&1}", LinearModel.Regressor.rust(nLabel, &1) |> elem(0)})

    rayon_result = nDatas
    |> Enum.map(& { "#{nLabel}, #{&1}", LinearModel.Regressor.part_rayon(nLabel, &1) |> elem(0)})
  end

  @doc """
  
  """
  def purelixir(nNum \\ 1, base \\ Params.ndata, offset \\ 1000, nLabel \\ Params.nlabel) do

  end

  @doc """
  
  """
  def elixir_inlining(nNum \\ 1, base \\ Params.ndata, offset \\ 1000, nLabel \\ Params.nlabel) do

  end

  @doc """

  """
  def rust(nNum \\ 1, base \\ Params.ndata, offset \\ 1000, nLabel \\ Params.nlabel) do
    LinearRegressorNif.SingleCore.new(0, nNum)
    |> Enum.map(& &1*offset + base)
    |> Enum.map(& { "#{nLabel}, #{&1}", LinearModel.Regressor.rust(nLabel, &1) |> elem(0)})
  end
  
  @doc """

  """
  def part_rayon(nNum \\ 1, base \\ Params.ndata, offset \\ 1000, nLabel \\ Params.nlabel) do
    LinearRegressorNif.SingleCore.new(0, nNum)
    |> Enum.map(& &1*offset + base)
    |> Enum.map(& { "#{nLabel}, #{&1}", LinearModel.Regressor.part_rayon(nLabel, &1) |> elem(0)})
  end

  @doc """

  """
  def filled_rayon(nNum \\ 1, base \\ Params.ndata, offset \\ 1000, nLabel \\ Params.nlabel) do
    LinearRegressorNif.SingleCore.new(0, nNum)
    |> Enum.map(& &1*offset + base)
    |> Enum.map(& { "#{nLabel}, #{&1}", LinearModel.Regressor.filled_rayon(nLabel, &1) |> elem(0)})
  end
end