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
  
  # Multi Core CPU
  def fit_little_rayon(_x, _y, _alpha, _iteration), do: exit(:nif_not_loaded)
  def fit_filled_rayon(_x, _y, _alpha, _iteration), do: exit(:nif_not_loaded)

  # GPGPU
  def gpuinfo(), do: exit(:nif_not_loaded)
  def gpu_dot_product(_, _), do: exit(:nif_not_loaded)

  # benchmark
  def benchmark_filled_rayon(_x, _y, _alpha, _iteration), do: exit(:nif_not_loaded)
  def benchmark_little_rayon(_x, _y, _alpha, _iteration), do: exit(:nif_not_loaded)
end
