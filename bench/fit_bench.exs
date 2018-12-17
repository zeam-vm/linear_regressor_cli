defmodule PseudoModelBench do
  use Benchfella

  before_each_bench _ do
    train_data = LinearModel.setup(13, 506)
    {:ok, train_data}
  end

  bench "Rust", [data: bench_context] do
    LinearModel.rust_regressor(data)
  end

  bench "Rayon", [data: bench_context] do
    LinearModel.rayon_regressor(data)
  end

  bench "Rayon variable thread",  [data: bench_context] do
    LinearModel.variable_thread_benchmark(data)
  end

end