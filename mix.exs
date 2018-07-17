defmodule LinearRegressorCli.MixProject do
  use Mix.Project

  def project do
    [
      app: :linear_regressor_cli,
      version: "0.1.0",
      elixir: "~> 1.6",
      start_permanent: Mix.env() == :prod,
      deps: deps()
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
    ]
  end
end
