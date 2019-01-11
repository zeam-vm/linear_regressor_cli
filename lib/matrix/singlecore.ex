defmodule LinearRegressorNif.SingleCore do
  import List.Util

  defmacro is_integer(a, b) do
    quote do
      unquote(a) |> is_integer and unquote(b) |> is_integer
    end
  end

  # Wrapper
  def new(a, b)
    when is_integer(a, b) do
    LinearRegressorNif.new(a, b)
    receive do
      l -> l
    end
  end


  def dot_product(a, b)
    when is_list(a) and is_list(b) do
      a
      |> to_float
      |> LinearRegressorNif.dot_product( b |> to_float )
  end

  def fit( x, y, alpha, iterations ) do
    LinearRegressorNif.fit(
      x , 
      y , 
      alpha ,
      iterations)
    receive do
      l -> l
    end
  end

end
