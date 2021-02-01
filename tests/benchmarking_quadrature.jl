using BenchmarkTools
using StaticArrays

include("../src/mesh.jl")
include("../src/quadrature.jl")

n = 7

scale_factor = 1.1
l2Norm(x, y, z) = im*sqrt(x^2 + y^2 + z^2)
points = hcat(randn(n), randn(n), randn(n))
spoints = SArray{Tuple{n, 3}}(points)
stpoints = SArray{Tuple{3, n}}(transpose(points))
weights = randn(n)
sweights = SVector{n}(weights)
srule = SArray{Tuple{4, n}}(transpose(hcat(points, weights)))
nodes = [1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0]


# integrateTriangle(nodes, l2Norm, points, weights)
# gaussQuadrature(scale_factor, l2Norm, points, weights)


function gaussQuadrature2(scale_factor, func::Function, rule::AbstractArray{Float64, 2})
    pnt_idxs = 1:3
    weight_idx = 4
    num_points = size(rule)[2]
    @views x, y, z = rule[pnt_idxs, 1]
    quadrature_sum = rule[weight_idx, 1] * func(x, y, z) # taken outside the loop to avoid type conversions/ambiguity
    if num_points > 1
        for sum_idx in 2:num_points
            @views x, y, z = rule[pnt_idxs, sum_idx]
            quadrature_sum += rule[weight_idx, sum_idx] * func(x, y, z)
        end
    end
    scale_factor * quadrature_sum
end


function integrateTriangle3(nodes::Array{Float64, 2}, func::Function, quadrature_rule::AbstractArray{Float64, 2})
    num_points = size(quadrature_rule)[2]
    triangle_area = norm(cross(nodes[2,:]-nodes[1,:], nodes[3,:]-nodes[2,:]))/2.0
    func_quadrature_rule = Array{Float64, 2}(undef, 4, num_points)
    @views func_quadrature_rule[4, :] = quadrature_rule[4, :]
    for point_idx in 1:num_points
        @views func_quadrature_rule[1:3, point_idx] = barycentric2Cartesian(nodes, quadrature_rule[1:3, point_idx])
    end
    gaussQuadrature2(triangle_area, func, func_quadrature_rule)
end
