defmodule LinearRegressorNif do
  use Rustler, otp_app: :linear_regressor_cli, crate: :linear_regressor_nif

  @doc """
  ## Examples

  """

  # For List Function
  def _dot_product(_a, _b), do: exit(:nif_not_loaded)
  def _zeros(_a), do: exit(:nif_not_loaded)
  def _new(_a, _b), do: exit(:nif_not_loaded)
  def _sub(_a, _b), do: exit(:nif_not_loaded)
  def _emult(_a, _b), do: exit(:nif_not_loaded)
  def _call_ocl(_a, _b), do: exit(:nif_not_loaded)
  def gpuinfo(), do: exit(:nif_not_loaded)

  # Main Function
  def _fit(_x, _y, _theta, _alpha, _iteration), do: exit(:nif_not_loaded)

  # Wrapper
  def dot_product(a, b)
      when is_list(a) and is_list(b) do
    _dot_product(
      a |> to_float,
      b |> to_float
    )
    |> IO.puts()
  end

  def ocl_dp(a, b)
      when is_list(a) and is_list(b) do
    _call_ocl(
      a |> to_float,
      b |> to_float
    )

    receive do
      l ->
        l
        |> IO.puts()
    end
  end

  # Sub Function
  def to_float(num) when is_integer(num), do: num / 1

  def to_float(r) when is_list(r) do
    r
    |> Enum.map(&to_float(&1))
  end

  def to_float(any), do: any

  def fit(x, y, theta, alpha, iterations) do
    # _fit(
    #   x |> to_float, 
    #   y |> to_float, 
    #   theta |> to_float, 
    #   alpha |> to_float,
    #   iterations)

    _fit(
      x,
      y,
      theta,
      alpha,
      iterations
    )

    receive do
      l -> l
    end
  end

  def test_dp() do
    m = 0..10_000_000 |> Enum.to_list()

    :timer.tc(fn -> Matrix.dot_product(m, m) end)
    |> elem(0)
    |> Kernel./(1_000_000)
    |> IO.puts()
  end

  def test_dp_rust() do
    m = 0..10_000_000 |> Enum.to_list()

    :timer.tc(fn -> dot_product(m, m) end)
    |> elem(0)
    |> Kernel./(1_000_000)
    |> IO.puts()
  end

  def test_ocl_dp do
    m = 0..10_000_000 |> Enum.to_list()

    :timer.tc(fn -> ocl_dp(m, m) end)
    |> elem(0)
    |> Kernel./(1_000_000)
    |> IO.puts()
  end
end


