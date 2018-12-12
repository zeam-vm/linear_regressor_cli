defmodule Benchmark do
  defmacro time(exp) do
    quote do
      {time, dict} = :timer.tc(fn() -> unquote(exp) end)
      IO.inspect "time: #{time} micro second"
      IO.inspect "-------------"
      dict
    end
  end

  def pseudo_linear_model(nLabel \\ 13, nData \\ 506) do
    LinearModel.run
    |> time
  end

end