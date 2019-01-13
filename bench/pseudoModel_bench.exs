defmodule PseudoModelBench do
  use Benchfella

  bench "Speed up efficiency", [] do
    LinearModel.all_benchmark 10
  end
end
