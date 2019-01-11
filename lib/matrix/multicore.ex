defmodule LinearRegressorNif.MultiCore do
  def fit( x, y, alpha, iterations ) do
    LinearRegressorNif.rayon_fit(
      x , 
      y , 
      alpha ,
      iterations)
    receive do
      l -> l
    end
  end

  def benchmark( x, y, alpha, iterations ) do
    LinearRegressorNif.benchmark(
      x , 
      y , 
      alpha ,
      iterations)
    receive do
      l -> l
    end
  end

end
