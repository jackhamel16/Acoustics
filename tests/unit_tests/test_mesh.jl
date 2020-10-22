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
        @test typeof(reshapeMeshArray([1,2,3,4,5,6], 2, Float64)) == Array{Float64, 2}
        @test typeof(reshapeMeshArray([1.,2.,3.,4.,5.,6.], 3)) == typeof([i*3.0+j for i in 0:1, j in 1:3])
    end
    @testset "buildPulseMesh tests" begin
        num_elements = 2
        num_coord_dims_solution = 3
        nodes_per_triangle_solution = 3
        nodes_solution = [0.0 0.0 0.0; 0.0 1.0 0.0; 1.0 1.0 0.0; 1.0 0.0 0.0]
        elements_solution = [2 1 4; 2 4 3]
        centroids_solution = [1/3 1/3 0.0; 2/3 2/3 0.0]

        test_mesh_filename = "examples/simple/rectangle_plate.msh"
        testPulseMesh = buildPulseMesh(test_mesh_filename)

        @test testPulseMesh.num_elements == num_elements
        @test testPulseMesh.num_coord_dims == num_coord_dims_solution
        @test testPulseMesh.nodes_per_triangle == nodes_per_triangle_solution
        @test testPulseMesh.nodes == nodes_solution
        @test testPulseMesh.elements == elements_solution
        @test testPulseMesh.centroids == centroids_solution
    end
    @testset "barycentric2Cartesian tests" begin
        nodes = [0.0 0.0 0.0; 0.0 1.0 1.0; 1.0 0.0 1.0]
        barycentric_coords = [1.0, 0.0, 0.0]
        cartesian_coords = [0.0, 0.0, 0.0]
        @test barycentric2Cartesian(nodes, barycentric_coords) == cartesian_coords
        barycentric_coords = [0.5, 0.5, 0.0]
        cartesian_coords = [0.0, 0.5, 0.5]
        @test barycentric2Cartesian(nodes, barycentric_coords) == cartesian_coords
        nodes = [0.0 -1.0 1.0; -2.0 0.0 0.0; 2.0 0.0 -0.5]
        barycentric_coords = [0.6, 0.6, -0.2]
        cartesian_coords = [-1.6, -0.6, 0.7]
        @test barycentric2Cartesian(nodes, barycentric_coords) == cartesian_coords
    end
end
