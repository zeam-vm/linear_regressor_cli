defmodule LinearRegressorMatrex do
    def predict( x, theta ), do: Matrex.multiply( x, theta )
    def cost( x, y, theta ) do
        m = length( y )
        hx_y = x 
            |> Matrex.multiply( theta )
            |> Matrex.subtract( y )
        hx_y_2 = Matrex.multiply( hx_y, hx_y )
        sum = hx_y_2
            |> Enum.reduce( 0, fn( x, acc ) -> Enum.at( x, 0 ) + acc end )
        sum / ( m * 2 )
    end
    def fit( x, y, theta, alpha, iterations ) do
        0..iterations |> Enum.to_list |> Enum.reduce( theta, fn( _iteration, theta ) -> 
            m = length( y )
            d = Matrex.multiply( Matrex.transpose( x ), Matrex.subtract( Matrex.multiply( x, theta ), y ) )
            size = Matrex.size( d )
            a = Matrex.new( elem( size, 0 ), elem( size, 1 ), alpha * ( 1 / m ) )
            d = Matrex.multiply( d, a )
            Matrex.subtract( theta, d )
        end )
    end
end
