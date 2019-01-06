defmodule LinearRegressorNif.SingleCore do
  import List.Util

  # Wrapper
  def dot_product(a, b)
    when is_list(a) and is_list(b) do
      a
      |> to_float
      |> LinearRegressorNif._dot_product( b |> to_float )
  end

  def fit( x, y, alpha, iterations ) do
    LinearRegressorNif._fit(
      x , 
      y , 
      alpha ,
      iterations)
    receive do
      l -> l
    end
  end

end
