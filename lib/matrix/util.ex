defmodule List.Util do
  # Sub Function
  def to_float(num) when is_integer(num), do: num /1
  def to_float(r) when is_list(r) do
    r
    |>Enum.map( &to_float(&1) )
  end
  def to_float(any), do: any
end