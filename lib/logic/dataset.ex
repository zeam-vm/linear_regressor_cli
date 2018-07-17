defmodule Dataset do
	def load_datas( path ) do
		datas = path |> load( [ headers: true ] )
		Map.keys( datas |> Enum.at( 0 ) )
		|> Enum.map( fn key -> 
			{
				key   |> String.downcase |> String.to_atom, 
				datas |> Enum.map( fn data -> data[ key ] end ) |> Enum.map( fn data -> elem( Float.parse( data ), 0 ) end )
			} 
		end )
	end

	def load( path, options ) when is_list( options ), do: load( path, options |> Enum.into( %{} ) )
	def load( path, %{ headers: _ } = options ) do
		File.stream!( path )
		|> CSV.decode( Enum.into( options, [] ) )
		|> Enum.to_list
		|> Enum.filter( &( elem( &1, 0 ) == :ok ) )
		|> Enum.map( &( elem( &1, 1 ) ) )
	end
end
