defmodule LinearRegressorNif do
  use Rustler, otp_app: :linear_regressor_cli, crate: :linear_regressor_nif

  @doc """
  ## Examples
  
  """

  # For List Function
  def _dot_product(_a, _b), do: exit(:nif_not_loaded)
  def _zeros(_a), do: exit(:nif_not_loaded)
  def _new(_a, _b), do: exit(:nif_not_loaded)
  def _sub(_a, _b), do: exit(:nif_not_loaded)
  def _emult(_a, _b), do: exit(:nif_not_loaded)

  # Main Function
  def _fit(_x, _y, _theta, _alpha, _iteration), do: exit(:nif_not_loaded)

  # Wrapper
  def dot_product(a, b)
    when is_list(a) and is_list(b) do
      a
      |> to_float
      |> _dot_product( b |> to_float )
  end

  # Sub Function
  def to_float(num) when is_integer(num), do: num /1
  def to_float(r) when is_list(r) do
    r
    |>Enum.map( &to_float(&1) )
  end
  def to_float(any), do: any
  
  def fit( x, y, theta, alpha, iterations ) do
    _fit(
      x |> to_float, 
      y |> to_float, 
      theta |> to_float, 
      alpha |> to_float,
      iterations)
    receive do
      l -> l
    end
  end

  # def fit_with_nif( x, y, theta, alpha, iterations ) do
  #   m = length( y )
  #   tx = Matrix.transpose( x )
  #   size = Matrix.size( theta )
  #   a = Matrix.new( elem( size, 0 ), elem( size, 1 ), alpha * ( 1 / m ) )

  #   0..iterations
  #     |> Enum.to_list
  #     |> Enum.reduce( theta, fn( _iteration, theta ) ->
  #       trans_theta = Matrix.transpose( theta )
  #       d = Enum.map(x, fn(row)->
  #         Enum.map(trans_theta, &dot_product(row, &1)) end)
  #       d = Enum.zip(d, y)
  #        |> Enum.map(fn({a,b})->sub(a,b)end)

  #       d = Matrix.transpose( d )
  #       d = Enum.map(d, fn(row) ->
  #           Enum.map(tx, &dot_product(row, &1))
  #       end)
  #       d = Matrix.transpose( d )
  #       d = Enum.zip(d, a)
  #         |> Enum.map( fn({a, b})->emult(a,b) end)
  #       Enum.zip(theta, d)
  #       |> Enum.map( fn({a,b})->sub(a,b) end)
  #     end)
  #   end
  # # end
end
