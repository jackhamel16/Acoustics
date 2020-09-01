using Test

include("../../src/element.jl")

@testset "element tests" begin
    @testset "compute_centroid tests" begin
        zero_vertices = zeros(3,3)
        @test compute_centroid(zero_vertices) == [0.,0.,0.]
        ones_vertices = ones(3,3)
        @test compute_centroid(ones_vertices) == [1,1,1]
        vertices=Array{Float64,2}(undef,3,3)
        for i in 1:3, j in 1:3
            vertices[i,j] = (2*i-1)*(j-2.5)
        end
        centroid = [-4.5, -1.5, 1.5]
        @test compute_centroid(vertices) == centroid
    end
    @testset "build_triangle_element tests" begin
    end
end
