defmodule LinearRegressorNif do
  use Rustler, otp_app: :linear_regressor_cli, crate: :linear_regressor_nif

  @doc """
  ## Examples
  
  """

  # List Function
  #  Single Core
  def dot_product(_a, _b), do: exit(:nif_not_loaded)
  def zeros(_a), do: exit(:nif_not_loaded)
  def new(_a, _b), do: exit(:nif_not_loaded)
  def sub(_a, _b), do: exit(:nif_not_loaded)
  def emult(_a, _b), do: exit(:nif_not_loaded)

  #  Multi Core
  def rayon_dot_product(_a, _b), do: exit(:nif_not_loaded)
  def rayon_sub(_a, _b), do: exit(:nif_not_loaded)
  def rayon_emult(_a, _b), do: exit(:nif_not_loaded)

  # Main Function
  #  Single Core
  def fit(_x, _y, _alpha, _iteration), do: exit(:nif_not_loaded)
  
  # Multi Core
  def rayon_fit(_x, _y, _alpha, _iteration), do: exit(:nif_not_loaded)
  
  def benchmark(_x, _y, _alpha, _iteration), do: exit(:nif_not_loaded)
end
