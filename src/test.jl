using LinearAlgebra
using IterativeSolvers
using LinearMaps

function identityOperator(x::AbstractVector)
    vec_size = length(x)
    identity_matrix = I(vec_size)
    return(identity_matrix * x)
end

map1 = LinearMap(identityOperator, 3)


matrix = [1 54 5; 0 0 -10; 3 18 -30]

function operator(x::AbstractVector)
    matrix = [1 54 5; 0 0 -10; 3 18 -30]
    return(matrix * x)
end

map2 = LinearMap(operator, 3)

x = randn(3)
rhs = matrix * x

test_sol = gmres(map2, rhs)

println("Results = ", isapprox(test_sol, x, rtol=1e-10))

for node_idx = 1:length(octree.nodes)
    node = octree.nodes[node_idx]
    println("node ", node_idx, " is child = ", isempty(node.children_idxs))
    println("    x bounds: ", node.bounds[1])
    println("    y bounds: ", node.bounds[2])
    println("    z bounds: ", node.bounds[3])
end
