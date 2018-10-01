defmodule LinearRegressor do
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
		0..iterations |> Enum.to_list |> Enum.reduce( theta, fn( _iteration, theta ) -> 
			m = length( y )
			d = Matrix.mult( Matrix.transpose( x ), Matrix.sub( Matrix.mult( x, theta ), y ) )
			size = Matrix.size( d ) # tuple
			# row...elem(size, 0), col...elem(size, 1)
			# a is 3row 1col
			a = Matrix.new( elem( size, 0 ), elem( size, 1 ), alpha * ( 1 / m ) )
			d = Matrix.emult( d, a )
			Matrix.sub( theta, d )
		end )
	end
end
