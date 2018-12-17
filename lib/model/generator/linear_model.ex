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
    # theta = List.duplicate([0.0], nLabel)

    {x_train, y_train, alpha, iterations}
  end

  def rust_fit({x_train, y_train, alpha, iterations}) do
    LinearRegressorNif.rust_fit( 
      x_train |> Matrix.transpose, 
      y_train |> Matrix.transpose, 
      alpha, 
      iterations )
  end

  def rayon_fit({x_train, y_train, alpha, iterations}) do
    LinearRegressorNif.rayon_fit( 
      x_train |> Matrix.transpose, 
      y_train |> Matrix.transpose, 
      alpha, 
      iterations )
  end

  def to_int(num) when is_float(num) do
    num |> Float.floor |> Float.to_string |> Integer.parse |> elem(0)
  end

  def variable_thread_benchmark({x_train, y_train, alpha, iterations}) do
    LinearRegressorNif.nif_benchmark( 
      x_train |> Matrix.transpose, 
      y_train |> Matrix.transpose, 
      alpha, 
      iterations )
  end
end
