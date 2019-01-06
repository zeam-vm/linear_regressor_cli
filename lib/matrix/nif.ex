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
