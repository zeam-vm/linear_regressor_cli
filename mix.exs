defmodule LinearRegressorCli.MixProject do
  use Mix.Project

  def project do
    [
      app: :linear_regressor_cli,
      version: "0.1.0",
      elixir: "~> 1.6",
      compilers: [:rustler] ++ Mix.compilers,
      rustler_crates: rustler_crates(),
      start_permanent: Mix.env() == :prod,
      deps: deps()
    ]
  end

  defp rustler_crates() do
    [
      linear_regressor_nif: [
        path: "native/linear_regressor_nif",
        mode: :release,
      ]
    ]
  end

  # Run "mix help compile.app" to learn about applications.
  def application do
    [
      extra_applications: [:logger]
    ]
  end

  # Run "mix help deps" to learn about dependencies.
  defp deps do
    [
      { :matrix,    "~> 0.3.2" },
      { :csv,       "~> 2.0.0" },
      { :rustler,   "~> 0.18.0"},
      { :benchfella, "~> 0.3.0"},
    ]
  end
end
