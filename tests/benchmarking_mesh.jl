using BenchmarkTools

include("../src/mesh.jl")

# Only optimize barycventric2cartesian since it is the only funciton called repeatedly in the fill routines
n = 1

barycentric_coords = randn(3)
coords_view = view(barycentric_coords,:)
nodes = [1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0]

barycentric2Cartesian(nodes, barycentric_coords)
