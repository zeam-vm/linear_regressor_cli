defmodule Util do
	# Benchmark
	# tool
	def timer( f ) do
		:timer.tc( fn -> f.() end)
		|> case do
			{ elapsed, res } -> IO.inspect( elapsed / 1_000_000 )
			res
		end
	end

	### make list
	# later

	@index 10_000_000

	def zeros(n) when n < 1, do: []
	def zeros(n) do
		[0] ++ zeros(n-1)
	end

	def new(e1, e2) when e1 > e2, do: []
	def new(e1, e2) do
		[e1] ++ new(e1+1, e2)
	end

	def dot_product(r1, _r2) when r1 == [], do: 0
	def dot_product(r1, r2) do
		[h1|t1] = r1
		[h2|t2] = r2
		(h1*h2) + dot_product(t1, t2)
	end


	def new_list_benchmark do
		:timer.tc( fn ->
			1..@index
			|> Enum.to_list
		end)
		|>elem(0)
		|>Kernel./(1_000_000)
	end

	# faster
	def new_list_benchmark1 do
		:timer.tc( fn ->
			NifRegressor.new(1, @index) 
		end)
		|>elem(0)
		|>Kernel./(1_000_000)
	end

	def new_list_benchmark2 do
		:timer.tc( fn ->
			new(1, @index) 
		end)
		|>elem(0)
		|>Kernel./(1_000_000)
	end

	### zeros 
	# very fast
	def zeros_benchmark do
		:timer.tc( fn ->
			List.duplicate(0, @index)
		end)
		|>elem(0)
		|>Kernel./(1_000_000)
	end 

	# later 
	def zeros_benchmark1 do
		:timer.tc( fn ->
			zeros(@index)
		end)
		|>elem(0)
		|>Kernel./(1_000_000)
	end

	# fast
	def zeros_benchmark2 do
		:timer.tc( fn ->
			NifRegressor.zeros(@index)
		end)
		|>elem(0)
		|>Kernel./(1_000_000)
	end

	def dp_benchmark do
		m = NifRegressor.new(1, @index) 
		:timer.tc( fn ->
			dot_product(m, m) 
		end)
		|>elem(0)
		|>Kernel./(1_000_000)
	end

	def dp_benchmark1 do
		m = NifRegressor.new(1, @index)
		:timer.tc( fn ->
			dot_product(m, m)
		end)
		|>elem(0)
		|>Kernel./(1_000_000)
	end
end