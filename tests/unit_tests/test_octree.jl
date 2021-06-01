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
    @testset "computeNodeBounds tests" begin
        centroid = [0.0,0.0,0.0]
        half_edge_length = 0.5
        sol_bounds_xyz = (-0.5,0.5)
        test_bounds = computeNodeBounds(half_edge_length, centroid)
        @test isapprox(test_bounds[1][1], sol_bounds_xyz[1], rtol=1e-15)
        @test isapprox(test_bounds[1][2], sol_bounds_xyz[2], rtol=1e-15)
        @test isapprox(test_bounds[2][1], sol_bounds_xyz[1], rtol=1e-15)
        @test isapprox(test_bounds[2][2], sol_bounds_xyz[2], rtol=1e-15)
        @test isapprox(test_bounds[3][1], sol_bounds_xyz[1], rtol=1e-15)
        @test isapprox(test_bounds[3][2], sol_bounds_xyz[2], rtol=1e-15)

        centroid = [-0.5,0.0,1.0]
        half_edge_length = 2.0
        sol_bounds = [(-2.5,1.5),(-2,2),(-1,3)]
        test_bounds = computeNodeBounds(half_edge_length, centroid)
        @test isapprox(test_bounds[1][1], sol_bounds[1][1], rtol=1e-15)
        @test isapprox(test_bounds[1][2], sol_bounds[1][2], rtol=1e-15)
        @test isapprox(test_bounds[2][1], sol_bounds[2][1], rtol=1e-15)
        @test isapprox(test_bounds[2][2], sol_bounds[2][2], rtol=1e-15)
        @test isapprox(test_bounds[3][1], sol_bounds[3][1], rtol=1e-15)
        @test isapprox(test_bounds[3][2], sol_bounds[3][2], rtol=1e-15)
    end # computeNodeBounds tests
    @testset "initializeOctree tests" begin
        num_levels = 1
        ele_centroids = [[1.0,1,0],[-1.0,-1,0]]
        node_centroid =[0.0,0.0,0.0]
        box_bounds = [-sqrt(2),sqrt(2)] # same for all dimensions
        sol_top_node_idx = 1
        sol_num_nodes = 1
        test_octree = initializeOctree(num_levels, ele_centroids)
        test_node = test_octree.nodes[1]
        @test isapprox(test_octree.num_levels, num_levels, rtol=1e-15)
        @test isapprox(test_octree.top_node_idx, sol_top_node_idx, rtol=1e-15)
        @test isapprox(length(test_octree.nodes), sol_num_nodes, rtol=1e-15)
        @test isapprox(test_node.centroid, node_centroid, rtol=1e-15)
        @test isapprox(test_node.parent_idx, 0, rtol=1e-15)
        @test isempty(test_node.children_idxs)
        @test isapprox(test_node.bounds[1][1], box_bounds[1], rtol=1e-15)
        @test isapprox(test_node.bounds[1][2], box_bounds[2], rtol=1e-15)
        @test isapprox(test_node.bounds[2][1], box_bounds[1], rtol=1e-15)
        @test isapprox(test_node.bounds[2][2], box_bounds[2], rtol=1e-15)
        @test isapprox(test_node.bounds[3][1], box_bounds[1], rtol=1e-15)
        @test isapprox(test_node.bounds[3][2], box_bounds[2], rtol=1e-15)
        #
        num_levels = 3
        ele_centroids = [[-1.0,0.2,0.0],[0.0,1.5,0.3],[0.0,0.0,1.0]]
        node_centroid = [-1,1.7,1.3] ./ 3
        max_ele_distance = 1.0
        sol_x_bounds = [-1/3-max_ele_distance,-1/3+max_ele_distance]
        sol_top_node_idx = 1
        sol_num_nodes = 1
        test_octree = initializeOctree(num_levels, ele_centroids)
        test_node = test_octree.nodes[1]
        @test isapprox(test_octree.num_levels, num_levels, rtol=1e-15)
        @test isapprox(test_octree.top_node_idx, sol_top_node_idx, rtol=1e-15)
        @test isapprox(length(test_octree.nodes), sol_num_nodes, rtol=1e-15)
        @test isapprox(test_node.centroid, node_centroid, rtol=1e-15)
        @test isapprox(test_node.parent_idx, 0, rtol=1e-15)
        @test isempty(test_node.children_idxs)
        @test isapprox(test_node.bounds[1], sol_x_bounds, rtol=1e-15)
    end # initializeOctree tests
end # octree tests
