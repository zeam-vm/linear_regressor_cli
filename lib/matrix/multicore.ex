defmodule LinearRegressorNif.MultiCore do
  import List.Util

  def dot_product(a, b)
    when is_list(a) and is_list(b) do
      LinearRegressorNif.rayon_dot_product(a |> to_float, b |> to_float )
      |> IO.inspect(label: "Answer")
  end
  
  def fit( x, y, alpha, iterations ) do
    LinearRegressorNif.fit_little_rayon(
      x , 
      y , 
      alpha ,
      iterations)
    receive do
      l -> l
    end
  end

  def fit_filled_rayon( x, y, alpha, iterations ) do
    LinearRegressorNif.fit_filled_rayon(
      x , 
      y , 
      alpha ,
      iterations)
    receive do
      l -> l
    end
  end

  def benchmark_filled_rayon( x, y, alpha, iterations ) do
    LinearRegressorNif.benchmark_filled_rayon(
      x , 
      y , 
      alpha ,
      iterations)
    receive do
      l -> l
    end
  end

  def benchmark_little_rayon( x, y, alpha, iterations ) do
    LinearRegressorNif.benchmark_little_rayon(
      x , 
      y , 
      alpha ,
      iterations)
    receive do
      l -> l
    end
  end

end
