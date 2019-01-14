defmodule LinearRegressorNif.MultiCore do
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
