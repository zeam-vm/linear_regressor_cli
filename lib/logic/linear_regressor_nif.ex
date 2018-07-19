defmodule LinearRegressorNif do
	use Rustler, otp_app: :linear_regressor_cli, crate: :linear_regressor_nif


 	@doc """
 	add.

	## Examples

	iex> LinearRegressorNif.add(1, 2)
	{:ok, 3}
	"""
	def add(_a, _b), do: exit(:nif_not_loaded)


	def predict( x, theta ), do: Matrix.mult( x, theta )

	def cost( x, y, theta ) do
		m = length( y )

		hx_y = x 
			|> Matrix.mult( theta )
			|> Matrix.sub( y )

		hx_y_2 = Matrix.emult( hx_y, hx_y )

		sum = hx_y_2
			|> Enum.reduce( 0, fn( x, acc ) -> Enum.at( x, 0 ) + acc end )

		sum / ( m * 2 )
	end

    def fit( x, y, theta, alpha, iterations ) do
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
            |> Enum.map(fn({a,b})->subtract_rows(a,b) end)

            d = Matrix.transpose( d )
            d = Enum.map(d, fn(row) ->
            		Enum.map(tx, &dot_product(row, &1))
            	end)
            d = Matrix.transpose( d )
            d = Enum.zip(d, a)
            	|> Enum.map( fn({a, b})->emult_rows(a,b) end)
            Enum.zip(theta, d)
            |> Enum.map( fn({a,b})->subtract_rows(a,b) end)
        end )
    end

 	@doc """
 	subtract_rows.

	## Examples

	iex> LinearRegressorNif.subtract_rows([4, 5, 6], [1, 2, 3])
	[3, 3, 3]

	iex> LinearRegressorNif.subtract_rows([4.0, 5.0, 6.0], [1.0, 2.0, 3.0])
	[3.0, 3.0, 3.0]

	"""
	def subtract_rows(r1, r2) when r1 == []  or  r2 == [], do: []
	def subtract_rows(r1, r2) do
		[h1|t1] = r1
		[h2|t2] = r2
		[h1-h2] ++ subtract_rows(t1,t2)
	end

 	@doc """
 	dot_product.

	## Examples

	iex> LinearRegressorNif.dot_product([1, 2, 3], [4, 5, 6])
	32

	iex> LinearRegressorNif.dot_product([1.0, 2.0, 3.0], [4.0, 5.0, 6.0])
	32.0

	"""
    def dot_product(r1, _r2) when r1 == [], do: 0
  	def dot_product(r1, r2) do
    	[h1|t1] = r1
    	[h2|t2] = r2
    	(h1*h2) + dot_product(t1, t2)
	end


 	@doc """
 	dot_product.

	## Examples

	iex> LinearRegressorNif.emult_rows([1, 2, 3], [4, 5, 6])
	[4, 10, 18]

	iex> LinearRegressorNif.emult_rows([1.0, 2.0, 3.0], [4.0, 5.0, 6.0])
	[4.0, 10.0, 18.0]

	"""
	def emult_rows(r1, r2) when r1 == []  or  r2 == [], do: []
	def emult_rows(r1, r2) do
		[h1|t1] = r1
		[h2|t2] = r2
		[h1*h2] ++ emult_rows(t1,t2)
	end

end