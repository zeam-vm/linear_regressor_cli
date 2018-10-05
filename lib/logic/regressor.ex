defmodule NifRegressor do
	use Rustler, otp_app: :linear_regressor_cli, crate: :regressor

	@doc """
 	## Examples
	
	"""

	# For List Function
	def dot_product(_a, _b), do: exit(:nif_not_loaded)
	def zeros(_a), do: exit(:nif_not_loaded)
	def new(_a, _b), do: exit(:nif_not_loaded)
	def sub(_a, _b), do: exit(:nif_not_loaded)
	def emult(_a, _b), do: exit(:nif_not_loaded)

	# Main Function
	def fit(_x, _y, _theta, _alpha, _iteration), do: exit(:nif_not_loaded)
	def test(_x, _y), do: exit(:nif_not_loaded)

	# Sub Function
	# def elem()
	# def dot_product(r1, _r2) when r1 == [],do: 0
	# def dot_product(r1, r2), do: nif_dot_product(r1, r2)

	def fit_with_nif( x, y, theta, alpha, iterations ) do
		m = length( y )
        tx = Matrix.transpose( x )
        size = Matrix.size( theta )
        a = Matrix.new( elem( size, 0 ), elem( size, 1 ), alpha * ( 1 / m ) )

        0..iterations
        |> Enum.to_list
        |> Enum.reduce( theta, fn( _iteration, theta ) ->
            trans_theta = Matrix.transpose( theta )
            d = Enum.map(x, fn(row)->
            		Enum.map(trans_theta, &dot_product(row, &1))
            	end)
            d =	Enum.zip(d, y)
            |> Enum.map(fn({a,b})->sub(a,b) end)

            d = Matrix.transpose( d )
            d = Enum.map(d, fn(row) ->
            		Enum.map(tx, &dot_product(row, &1))
            	end)
            d = Matrix.transpose( d )
            d = Enum.zip(d, a)
            	|> Enum.map( fn({a, b})->emult(a,b) end)
            Enum.zip(theta, d)
            |> Enum.map( fn({a,b})->sub(a,b) end)
        end )
	end

	# end 
end
