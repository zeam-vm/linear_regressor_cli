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

  # Main Function
  def _fit(_x, _y, _alpha, _iteration), do: exit(:nif_not_loaded)
  def _rayon_fit(_x, _y, _alpha, _iteration), do: exit(:nif_not_loaded)
  def _nif_benchmark(_x, _y, _alpha, _iteration), do: exit(:nif_not_loaded)

  # Wrapper
  def dot_product(a, b)
    when is_list(a) and is_list(b) do
      a
      |> to_float
      |> _dot_product( b |> to_float )
  end

  # Sub Function
  def to_float(num) when is_integer(num), do: num /1
  def to_float(r) when is_list(r) do
    r
    |>Enum.map( &to_float(&1) )
  end
  def to_float(any), do: any
  
  def rust_fit( x, y, alpha, iterations ) do
    _fit(
      x , 
      y , 
      alpha ,
      iterations)
    receive do
      l -> l
    end
  end

  def rayon_fit( x, y, alpha, iterations ) do
    _rayon_fit(
      x , 
      y , 
      alpha ,
      iterations)
    receive do
      l -> l
    end
  end

  def nif_benchmark( x, y, alpha, iterations ) do
    _nif_benchmark(
      x , 
      y , 
      alpha ,
      iterations)
    receive do
      l -> l
    end
  end
end
