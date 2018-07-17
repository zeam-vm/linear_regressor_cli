defmodule LinearRegressorCliTest do
  use ExUnit.Case
  doctest LinearRegressorCli

  test "greets the world" do
    assert LinearRegressorCli.hello() == :world
  end
end
