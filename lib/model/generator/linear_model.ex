defmodule LinearModel do

  @doc"""

  """

  # y = x + noize
  # 0 <= x < 100
  def generator (len \\ 10) do
    List.duplicate(100, len) # [100, 100, ... 100]
    |> Enum.map(& &1*:rand.uniform + :rand.uniform)
    |> Enum.sort
  end
end
