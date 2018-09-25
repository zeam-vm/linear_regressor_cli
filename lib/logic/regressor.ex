defmodule NifRegressor do
	use Rustler, otp_app: :linear_regressor_cli, crate: :regressor

	# For List Function
	def sum_list(_a), do: exit(:nif_not_loaded)
	def nif_dot_product(_a, _b), do: exit(:nif_not_loaded)
	def nif_zeros(_a), do: exit(:nif_not_loaded)
	def nif_new(_a, _b), do: exit(:nif_not_loaded)

	# def benchmark do
	# 	:timer.tc( Matrix.new()

	# 		)
	# end

	def zeros(n) when n < 1, do: []
	def zeros(n) do
		[0] ++ zeros(n-1)
	end

	def new(e1, e2) when e1 > e2, do: []
	def new(e1, e2) do
		[e1] ++ new(e1+1, e2)
	end

	# Benchmark
	#	none = original
	# 	1 = my function for elixir
	#	2 = my function for rustler 

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
	def new_list_benchmark do
		:timer.tc( fn ->
			1..10_000_000
			|> Enum.to_list
		end)
		|>elem(0)
		|>Kernel./(1_000_000)
	end

	# faster
	def new_list_benchmark1 do
		:timer.tc( fn ->
			new(1, 10_000_000) 
		end)
		|>elem(0)
		|>Kernel./(1_000_000)
	end

	### zeros 
	# very fast
	def zeros_benchmark do
		:timer.tc( fn ->
			List.duplicate(0, 10_000_000)
		end)
		|>elem(0)
		|>Kernel./(1_000_000)
	end 

	# later 
	def zeros_benchmark1 do
		:timer.tc( fn ->
			zeros(10_000_000)
		end)
		|>elem(0)
		|>Kernel./(1_000_000)
	end

	# fast
	def zeros_benchmark2 do
		:timer.tc( fn ->
			nif_zeros(10_000_000)
		end)
		|>elem(0)
		|>Kernel./(1_000_000)
	end

	def dp_benchmark do
		m = new(1, 10_000_000)
		:timer.tc( fn ->
			dot_product_ex(m, m) 
		end)
		|>elem(0)
		|>Kernel./(1_000_000)
	end

	def dp_benchmark1 do
		m = new(1, 10_000_000)
		:timer.tc( fn ->
			nif_dot_product(m, m)
		end)
		|>elem(0)
		|>Kernel./(1_000_000)
	end

	@doc """
 	dot_product.

	## Examples

	iex> LinearRegressorNif.dot_product([1, 2, 3], [4, 5, 6])
	32

	iex> LinearRegressorNif.dot_product([1.0, 2.0, 3.0], [4.0, 5.0, 6.0])
	32.0

	"""
	def dot_product_ex(r1, _r2) when r1 == [], do: 0
	def dot_product_ex(r1, r2) do
		[h1|t1] = r1
		[h2|t2] = r2
		(h1*h2) + dot_product_ex(t1, t2)
	end
end
