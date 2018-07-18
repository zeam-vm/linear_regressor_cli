defmodule BostonMatrex do
    def run do
        features = Dataset.load_datas( "data/boston_house_prices_x.csv" )
        targets  = Dataset.load_datas( "data/boston_house_prices_y.csv" )
        x_train = 
        [ 
            features[ :crim ], 
            features[ :zn ], 
            features[ :indus ], 
            features[ :chas ], 
            features[ :nox ], 
            features[ :rm ], 
            features[ :age ], 
            features[ :dis ], 
            features[ :rad ], 
            features[ :tax ], 
            features[ :ptratio ], 
            features[ :b ], 
            features[ :lstat ], 
        ]
        |> Matrex.transpose
        y_train = [ targets[ :medv ] ] |> Matrex.transpose
        alpha = 0.0000003
        iterations = 10000
        theta = [ [ 0 ], [ 0 ], [ 0 ], [ 0 ], [ 0 ], [ 0 ], [ 0 ], [ 0 ], [ 0 ], [ 0 ], [ 0 ], [ 0 ], [ 0 ] ]
        { spent, theta } = :timer.tc( fn -> 
            LinearRegressorMatrex.fit( x_train, y_train, theta, alpha, iterations ) end )
IO.puts "==========================="
IO.puts spent
IO.puts "==========================="
        x_test = [ [ 0.00632 ], [ 18.0 ], [ 2.31 ], [ 0.0 ], [ 0.538 ], [ 6.575 ], [ 65.2 ], [ 4.09 ], [ 1.0 ], [ 296.0 ], [ 15.3 ], [ 396.9 ], [ 4.98 ] ] |> Matrix.transpose
        y_test = [ [ 24.0 ] ] 
        predicted = LinearRegressorMatrex.predict( x_test, theta )
        error = LinearRegressorMatrex.cost( x_test, y_test, theta )
        IO.puts "y_test:  #{ y_test    |> inspect }"
        IO.puts "predict: #{ predicted |> inspect }"
        IO.puts ""
        IO.puts "error: "
        IO.inspect error
    end

    def benchmark do
        IO.puts (
            :timer.tc(fn -> run() end)
            |> elem(0)
            |> Kernel./(1000000)
        )
    end
end
