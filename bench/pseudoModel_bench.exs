defmodule PseudoModelBench do
  use Benchfella

  bench "Rust", [data: setup()] do
    {x_train, y_train, alpha, iterations} = data

    LinearRegressorNif.SingleCore.fit( 
      x_train , 
      y_train , 
      alpha, 
      iterations )
  end

  bench "Rayon", [data: setup()] do
    {x_train, y_train, alpha, iterations} = data

    LinearRegressorNif.MultiCore.fit( 
      x_train , 
      y_train , 
      alpha, 
      iterations )
  end

  defp setup do
    LinearModel.setup 10, 5000
  end
end
