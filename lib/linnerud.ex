defmodule Linnerud do

	def run do
		features = Dataset.load_datas( "data/linnerud_exercise.csv" )
		targets  = Dataset.load_datas( "data/linnerud_physiological.csv" )

		x_train = 
		[ 
			features[ :chins ], 
			features[ :situps ], 
			features[ :jumps ], 
		]
		|> Matrix.transpose
		y_train = [ targets[ :weight ] ] |> Matrix.transpose

		alpha = 0.0000003
		iterations = 10000
		theta = [ [ 0 ], [ 0 ], [ 0 ] ]

		theta = LinearRegressor.fit( x_train, y_train, theta, alpha, iterations )

		x_test = [ [ 5 ], [ 162 ], [ 60 ] ] |> Matrix.transpose
		y_test = [ [ 191 ] ] 

		predicted = LinearRegressor.predict( x_test, theta )

		error = LinearRegressor.cost( x_test, y_test, theta )

		IO.puts "y_test:  #{ y_test    |> inspect }"
		IO.puts "predict: #{ predicted |> inspect }"
		IO.puts ""
		IO.puts "error: "
		IO.inspect error
	end

	def run1 do
		features = Dataset.load_datas( "data/linnerud_exercise.csv" )
		targets  = Dataset.load_datas( "data/linnerud_physiological.csv" )

		x_train = 
		[ 
			features[ :chins ], 
			features[ :situps ], 
			features[ :jumps ], 
		]
		|> Matrix.transpose
		y_train = [ targets[ :weight ] ] |> Matrix.transpose

		alpha = 0.0000003
		iterations = 10000
		theta = [ [ 0.0 ], [ 0.0 ], [ 0.0 ] ]

		theta = NifRegressor.fit( x_train, y_train, theta, alpha, iterations )

		x_test = [ [ 5 ], [ 162 ], [ 60 ] ] |> Matrix.transpose
		y_test = [ [ 191 ] ] 

		predicted = LinearRegressor.predict( x_test, theta )

		error = LinearRegressor.cost( x_test, y_test, theta )

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

	def benchmark1 do
		IO.puts (
			:timer.tc(fn -> run1() end)
			|> elem(0)
			|> Kernel./(1000000)
		)
	end

end
