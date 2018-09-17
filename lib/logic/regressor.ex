defmodule NifRegressor do
	use Rustler, otp_app: :linear_regressor_cli, crate: :regressor

	def add(_a, _b), do: exit(:nif_not_loaded)
	# def dot_product(_a, _b), do: exit(:nif_not_loaded)


	defp print_tuple(_a), do: exit(:nif_not_loaded)
	def test_tuple() do
		# 4要素タプルの呼び出し
		tuple = {:im_an_atom, 1.0, 1, "string"}
		print_tuple(tuple)
	end

	# For List Function
	def sum_list(_a), do: exit(:nif_not_loaded)
	def dot_product(_a, _b), do: exit(:nif_not_loaded)
	
	# Debug
	def test1() do
		dot_product([1, 2, 3], [4, 5, 6])
	end

	def test2() do
		dot_product([[1, 2], [3, 4]], [[5, 6], [5, 6]])
	end

end
