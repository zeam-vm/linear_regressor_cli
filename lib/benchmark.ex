defmodule Benchmark do
  defmacro time(exp, tmp \\ false) do
    quote do
      {time, dict} = :timer.tc(fn() -> unquote(exp) end)
      IO.inspect "time: #{time} micro second"
      IO.inspect "-------------"
      case unquote(tmp) do
        false -> dict
        true -> {time, dict}
      end
    end
  end
end