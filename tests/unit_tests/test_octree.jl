using Test

include("../../src/octree.jl")

@testset "octree tests" begin
    # num_levels = 1
    # ele_centroids = [[1.0,1,0],[-1.0,-1,0]]
    # sol_map = [[1,2]]
    # test_octree = initializeOctree(num_levels, ele_centroids)
    # @test isapprox(test_octree.box_to_element_map, sol_map, rtol=1e-15)
    #
    # num_levels = 1
    # ele_centroids = [[1.0,1,0],[-1.0,-1,0],[-1.0,1,0]]
    # sol_map = [[1,2,3]]
    # test_octree = initializeOctree(num_levels, ele_centroids)
    # @test isapprox(test_octree.box_to_element_map, sol_map, rtol=1e-15)

    # num_levels = 2
    # ele_centroids = [[1,1,0.1],[-1,-1,-0.1]]
    # sol_map = [[2],[],[],[],[],[],[],[8]]
    # sol_map = [[],[],[],[],[],[],[],[]]
    # test_octree = initializeOctree(num_levels, ele_centroids)
    # @test isapprox(test_octree.box_to_element_map, sol_map, rtol=1e-15)
    @testset "initializeOctree tests" begin
        num_levels = 1
        ele_centroids = [[1.0,1,0],[-1.0,-1,0]]
        box_centroids = [[[0,0,0]]]
        test_octree = initializeOctree(num_levels, ele_centroids)
        @test isapprox(test_octree.num_levels, num_levels, rtol=1e-15)
        @test isapprox(test_octree.box_centroids, box_centroids, rtol=1e-15)
    end # initializeOctree tests
end # octree tests
