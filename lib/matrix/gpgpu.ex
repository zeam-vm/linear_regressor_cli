defmodule LinearRegressorNif.GPU do
  import List.Util

  def dot_product(a, b)
    when is_list(a) and is_list(b) do
      LinearRegressorNif.gpu_dot_product(
        a |> to_float, 
        b |> to_float )

      receive do
        ans -> ans
      end
  end
end
