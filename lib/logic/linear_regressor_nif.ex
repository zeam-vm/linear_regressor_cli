defmodule LinearRegressorNif do
	use Rustler, otp_app: :linear_regressor_cli, crate: :linear_regressor_nif


 	@doc """
 	add.

	## Examples

	iex> LinearRegressorNif.add(1, 2)
    	{:ok, 3}
	"""
	def add(_a, _b), do: exit(:nif_not_loaded)

end