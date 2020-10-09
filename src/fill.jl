using LinearAlgebra

include("quadrature.jl")

function rhsFill(num_elements::Int64, elements::Array{Int64, 2}, nodes::Array{Float64, 2}, fieldFunc::Function, quadrature_rule::Array{Float64, 2})
    rhs = Array{Complex{Float64}, 1}(undef, num_elements)
    for element_idx in 1:num_elements
        triangle_nodes = Array{Float64, 2}(undef, 3, 3)
        for node_idx_local in 1:3
            node_idx_global = elements[element_idx, node_idx_local]
            triangle_nodes[node_idx_local,:] = nodes[node_idx_global,:]
        end
        rhs[element_idx] = integrateTriangle(triangle_nodes, fieldFunc, quadrature_rule[:,1:3], quadrature_rule[:,4])
    end
    rhs
end
