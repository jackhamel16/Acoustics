using BenchmarkTools

include("../src/mesh.jl")
include("../src/quadrature.jl")

n = 7

scale_factor = 1.1
l2Norm(x, y, z) = sqrt(x^2 + y^2 + z^2)
points = hcat(randn(n), randn(n), randn(n))
weights = randn(n)
nodes = [1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0]


# integrateTriangle(nodes l2Norm, points, weights)
# gaussQuadrature(scale_factor, l2Norm, points, weights)
