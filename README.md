# LinearRegressorCli

**TODO: Add description**

```bash
$ mix run -e "Linnerud.run"
y_test:  [[191]]
predict: [[165.69839452596122]]

error:
320.08561978195456
320.08561978195456
```

```bash
$ mix run -e "Boston.run"
y_test:  [[24.0]]
predict: [[25.26741625578046]]

error:
0.803171982708281
0.803171982708281
```

# Benchmark
```bash
$ mix run -e "LinearRegressorNif.ocl_dp [1,2], [3,4]"
Compiling NIF crate :linear_regressor_nif (native/linear_regressor_nif)...
    Finished release [optimized] target(s) in 1.67s
11.0

```

## Installation

If [available in Hex](https://hex.pm/docs/publish), the package can be installed
by adding `linear_regressor_cli` to your list of dependencies in `mix.exs`:

```elixir
def deps do
  [
    {:linear_regressor_cli, "~> 0.1.0"}
  ]
end
```

Documentation can be generated with [ExDoc](https://github.com/elixir-lang/ex_doc)
and published on [HexDocs](https://hexdocs.pm). Once published, the docs can
be found at [https://hexdocs.pm/linear_regressor_cli](https://hexdocs.pm/linear_regressor_cli).

