using Test

include("../../src/mesh.jl")

@testset "mesh tests" begin
    @testset "computeCentroid tests" begin
        zero_vertices = zeros(3,3)
        @test computeCentroid(zero_vertices) == [0.,0.,0.]
        ones_vertices = ones(3,3)
        @test computeCentroid(ones_vertices) == [1,1,1]
        vertices=Array{Float64,2}(undef,3,3)
        for i in 1:3, j in 1:3
            vertices[i,j] = (2*i-1)*(j-2.5)
        end
        centroid = [-4.5, -1.5, 1.5]
        @test computeCentroid(vertices) == centroid
    end
    @testset "reshapeMeshArray tests" begin
        @test reshapeMeshArray([1,2,3,4,5,6], 2) == [i*2+j for i in 0:2, j in 1:2]
        @test typeof(reshapeMeshArray([1.,2.,3.,4.,5.,6.], 3)) == typeof([i*3.0+j for i in 0:1, j in 1:3])
    end
    @testset "buildPulseMesh tests" begin
        num_elements = 2
        num_coord_dims_solution = 3
        nodes_per_triangle_solution = 3
        nodes_solution = convert(Array{Float64}, transpose(hcat([0,0,0],[0,1,0],[1,1,0],[1,0,0])))
        elements_solution = convert(Array{UInt64}, transpose(hcat([2,1,4],[2,4,3])))
        centroids_solution = convert(Array{Float64}, transpose(hcat([1/3,1/3,0],[2/3,2/3,0])))

        test_mesh_filename = "examples/simple/rectangle_plate.msh"
        testPulseMesh = buildPulseMesh(test_mesh_filename)

        @test testPulseMesh.num_coord_dims == num_coord_dims_solution
        @test testPulseMesh.nodes_per_triangle == nodes_per_triangle_solution
        @test testPulseMesh.nodes == nodes_solution
        @test testPulseMesh.elements == elements_solution
        @test testPulseMesh.centroids == centroids_solution
    end
end
