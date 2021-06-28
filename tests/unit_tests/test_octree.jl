using Test

include("../../src/mesh.jl")
include("../../src/quadrature.jl")
include("../../src/fill.jl")
include("../../src/greens_functions.jl")
include("../../src/ACA.jl")

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
    @testset "createChildren tests" begin
        no_child_idxs = []
        no_Z_matrices = []
        parent_level = 1
        ele_centroids = [[1,1,0.1],[-1,-1,-0.1]]
        parent_centroid = [0.0,0.0,0.0]
        max_distance = norm(ele_centroids[1])
        parent_node = Node(parent_level, 0,no_child_idxs,[1,2],computeNodeBounds(max_distance, parent_centroid),parent_centroid,no_Z_matrices)
        sol_parent_idx = 1
        sol_child_level = parent_level + 1
        sol_element_idxs = [[2],[1]]
        sol_child1_bounds = [[-max_distance,0],[-max_distance,0],[-max_distance,0]]
        sol_child2_bounds = [[0,max_distance],[0,max_distance],[0,max_distance]]
        sol_child1_centroid = parent_centroid .- max_distance/2
        sol_child2_centroid = parent_centroid .+ max_distance/2
        test_child_nodes = createChildren(sol_parent_idx, parent_node, ele_centroids)
        @test isapprox(length(test_child_nodes), 2, rtol=1e-15)
        @test isapprox(test_child_nodes[1].parent_idx, sol_parent_idx, rtol=1e-15)
        @test isapprox(test_child_nodes[2].parent_idx, sol_parent_idx, rtol=1e-15)
        @test isapprox(test_child_nodes[1].octree_level, sol_child_level, rtol=1e-15)
        @test isapprox(test_child_nodes[2].octree_level, sol_child_level, rtol=1e-15)
        @test isempty(test_child_nodes[1].children_idxs)
        @test isempty(test_child_nodes[2].children_idxs)
        @test isapprox(test_child_nodes[1].element_idxs, sol_element_idxs[1], rtol=1e-15)
        @test isapprox(test_child_nodes[2].element_idxs, sol_element_idxs[2], rtol=1e-15)
        @test isapprox(test_child_nodes[1].bounds, sol_child1_bounds, rtol=1e-15)
        @test isapprox(test_child_nodes[2].bounds, sol_child2_bounds, rtol=1e-15)
        @test isapprox(test_child_nodes[1].centroid, sol_child1_centroid, rtol=1e-15)
        @test isapprox(test_child_nodes[2].centroid, sol_child2_centroid, rtol=1e-15)

        parent_level = 2
        ele_centroids = [[0.0,-1.0,1.2],[0.0,-1.0,0.5],[1.0,0.5,2.0]]
        parent_centroid = [0.5,-0.5,1.0]
        max_distance = 1.5
        parent_node = Node(parent_level, 0,no_child_idxs,[1,2,3],computeNodeBounds(max_distance,parent_centroid),parent_centroid,no_Z_matrices)
        sol_parent_idx = 10
        sol_child_level = parent_level + 1
        sol_children_ele_idxs = [[2],[1],[3]]
        sol_child_bounds = [[[-1.0,0.5],[-2.0,-0.5],[-0.5,1.0]],
                             [[-1.0,0.5],[-2.0,-0.5],[1.0,2.5]],
                             [[0.5,2.0],[-0.5,1.0],[1.0,2.5]]]
        sol_child_centroids = [[-0.25,-1.25,0.25],[-0.25,-1.25,1.75],[1.25,0.25,1.75]]
        test_child_nodes = createChildren(sol_parent_idx, parent_node, ele_centroids)
        @test isapprox(test_child_nodes[1].parent_idx, sol_parent_idx, rtol=1e-15)
        @test isapprox(test_child_nodes[2].parent_idx, sol_parent_idx, rtol=1e-15)
        @test isapprox(test_child_nodes[3].parent_idx, sol_parent_idx, rtol=1e-15)
        @test isapprox(test_child_nodes[1].octree_level, sol_child_level, rtol=1e-15)
        @test isapprox(test_child_nodes[2].octree_level, sol_child_level, rtol=1e-15)
        @test isapprox(test_child_nodes[3].octree_level, sol_child_level, rtol=1e-15)
        @test isempty(test_child_nodes[1].children_idxs)
        @test isempty(test_child_nodes[2].children_idxs)
        @test isempty(test_child_nodes[3].children_idxs)
        @test isapprox(test_child_nodes[1].element_idxs, sol_children_ele_idxs[1], rtol=1e-15)
        @test isapprox(test_child_nodes[2].element_idxs, sol_children_ele_idxs[2], rtol=1e-15)
        @test isapprox(test_child_nodes[3].element_idxs, sol_children_ele_idxs[3], rtol=1e-15)
        @test isapprox(test_child_nodes[1].bounds, sol_child_bounds[1], rtol=1e-15)
        @test isapprox(test_child_nodes[2].bounds, sol_child_bounds[2], rtol=1e-15)
        @test isapprox(test_child_nodes[3].bounds, sol_child_bounds[3], rtol=1e-15)
        @test isapprox(test_child_nodes[1].centroid, sol_child_centroids[1], rtol=1e-15)
        @test isapprox(test_child_nodes[2].centroid, sol_child_centroids[2], rtol=1e-15)
        @test isapprox(test_child_nodes[3].centroid, sol_child_centroids[3], rtol=1e-15)
    end # createChildren tests
    @testset "createOctree tests" begin
        src_quadrature_rule = gauss7rule
        test_quadrature_rule = gauss7rule
        mesh_filename = "examples/test/rectangle_plate_8elements_symmetric.msh"
        pulse_mesh =  buildPulseMesh(mesh_filename, src_quadrature_rule, test_quadrature_rule)

        num_nodes = 1
        small_buffer = 1e-4
        sol_num_levels = 1
        sol_top_node_idx = 1
        sol_leaf_node_idxs = [1]
        test_octree = createOctree(sol_num_levels, pulse_mesh)
        @test isapprox(test_octree.num_levels, sol_num_levels, rtol=1e-15)
        @test isapprox(test_octree.top_node_idx, sol_top_node_idx, rtol=1e-15)
        @test isapprox(test_octree.leaf_node_idxs, sol_leaf_node_idxs, rtol=1e-15)
        @test isapprox(length(test_octree.nodes), num_nodes, rtol=1e-15)

        num_nodes = 3
        small_buffer = 1e-4
        sol_num_levels = 2
        ele_centroids = Array{Array{Float64,1},1}(undef, pulse_mesh.num_elements)
        for ele_idx = 1:pulse_mesh.num_elements
            ele_centroids[ele_idx] = computeCentroid(pulse_mesh.nodes[pulse_mesh.elements[ele_idx,:],:])
        end
        sol_octree = initializeOctree(sol_num_levels, small_buffer, ele_centroids)
        sol_top_node_idx = 1
        fillOctreeNodes!(sol_top_node_idx, sol_octree, ele_centroids)
        # test_octree = createOctree(sol_num_levels, ele_centroids)
        test_octree = createOctree(sol_num_levels, pulse_mesh)
        @test isapprox(test_octree.num_levels, sol_octree.num_levels, rtol=1e-15)
        @test isapprox(test_octree.top_node_idx, sol_octree.top_node_idx, rtol=1e-15)
        @test isapprox(test_octree.leaf_node_idxs, sol_octree.leaf_node_idxs, rtol=1e-15)
        for node_idx = 1:num_nodes
            @test isapprox(test_octree.nodes[node_idx].octree_level, sol_octree.nodes[node_idx].octree_level, rtol=1e-15)
            @test isapprox(test_octree.nodes[node_idx].parent_idx, sol_octree.nodes[node_idx].parent_idx, rtol=1e-15)
            @test isapprox(test_octree.nodes[node_idx].children_idxs, sol_octree.nodes[node_idx].children_idxs, rtol=1e-15)
            @test isapprox(test_octree.nodes[node_idx].element_idxs, sol_octree.nodes[node_idx].element_idxs, rtol=1e-15)
            @test isapprox(test_octree.nodes[node_idx].bounds, sol_octree.nodes[node_idx].bounds, rtol=1e-15)
            @test isapprox(test_octree.nodes[node_idx].centroid, sol_octree.nodes[node_idx].centroid, rtol=1e-15)
        end
    end
    @testset "fillOctree tests" begin
        level1 = 1; level2 = 2; level3 = 3; level4 = 4
        no_buffer = 0.0
        num_levels = 3
        ele_centroids = [[1,1,0.1],[-1,-1,-0.1]]
        sol_top_node_idx = 1
        sol_num_nodes = 5
        sol_element_idxs = [[1,2],[2],[1],[2],[1]]
        sol_leaf_node_idxs = [4,5]
        test_octree = initializeOctree(num_levels, no_buffer, ele_centroids)
        fillOctreeNodes!(level1, test_octree, ele_centroids)
        @test isapprox(test_octree.num_levels, num_levels, rtol=1e-15)
        @test isapprox(length(test_octree.nodes), sol_num_nodes, rtol=1e-15)
        @test isapprox(test_octree.nodes[1].children_idxs, [2,3], rtol=1e-15)
        test_child_nodes = test_octree.nodes[test_octree.nodes[1].children_idxs]
        @test isapprox(test_octree.nodes[1].octree_level, level1, rtol=1e-15)
        @test isapprox(test_octree.nodes[2].octree_level, level2, rtol=1e-15)
        @test isapprox(test_octree.nodes[3].octree_level, level2, rtol=1e-15)
        @test isapprox(test_octree.nodes[4].octree_level, level3, rtol=1e-15)
        @test isapprox(test_octree.nodes[5].octree_level, level3, rtol=1e-15)
        for node_idx = 1:sol_num_nodes
            @test isapprox(test_octree.nodes[node_idx].element_idxs, sol_element_idxs[node_idx], rtol=1e-15)
        end
        @test isapprox(test_octree.leaf_node_idxs, sol_leaf_node_idxs)

        num_levels = 4
        ele_centroids = [[1,1,0.1],[-1,-1,-0.1]]
        sol_top_node_idx = 1
        sol_num_nodes = 7
        sol_element_idxs = [[1,2],[2],[1],[2],[2],[1],[1]]
        sol_leaf_node_idxs = [5,7]
        test_octree = initializeOctree(num_levels, no_buffer, ele_centroids)
        fillOctreeNodes!(sol_top_node_idx, test_octree, ele_centroids)
        @test isapprox(test_octree.top_node_idx, sol_top_node_idx, rtol=1e-15)
        @test isapprox(test_octree.num_levels, num_levels, rtol=1e-15)
        @test isapprox(length(test_octree.nodes), sol_num_nodes, rtol=1e-15)
        @test isapprox(test_octree.nodes[1].children_idxs, [2,3], rtol=1e-15)
        @test isapprox(test_octree.nodes[1].octree_level, level1, rtol=1e-15)
        @test isapprox(test_octree.nodes[2].octree_level, level2, rtol=1e-15)
        @test isapprox(test_octree.nodes[3].octree_level, level2, rtol=1e-15)
        @test isapprox(test_octree.nodes[4].octree_level, level3, rtol=1e-15)
        @test isapprox(test_octree.nodes[5].octree_level, level4, rtol=1e-15)
        @test isapprox(test_octree.nodes[6].octree_level, level3, rtol=1e-15)
        @test isapprox(test_octree.nodes[7].octree_level, level4, rtol=1e-15)
        for node_idx = 1:sol_num_nodes
            @test isapprox(test_octree.nodes[node_idx].element_idxs, sol_element_idxs[node_idx], rtol=1e-15)
        end
        @test isapprox(test_octree.leaf_node_idxs, sol_leaf_node_idxs)
    end
    @testset "fillOctreeZMatricesSoundSoft! tests" begin
        wavenumber = 1.0+0.0im
        src_quadrature_rule = gauss7rule
        test_quadrature_rule = gauss7rule
        distance_to_edge_tol = 1e-12
        near_singular_tol = 1.0
        approximation_tol = 1e-3
        compression_distance = 1.5 # should exclude all box touching sides or corners
        ACA_tol = 1e-4
        mesh_filename = "examples/test/rectangle_plate_8elements_symmetric.msh"
        pulse_mesh =  buildPulseMesh(mesh_filename, src_quadrature_rule, test_quadrature_rule)
        testIntegrand(r_test, src_idx, is_singular) = scalarGreensIntegration(pulse_mesh, src_idx,
                                                       wavenumber,
                                                       r_test,
                                                       distance_to_edge_tol,
                                                       near_singular_tol,
                                                       is_singular)
        z_matrix = zeros(ComplexF64, pulse_mesh.num_elements, pulse_mesh.num_elements)
        matrixFill(pulse_mesh, testIntegrand, z_matrix)
        num_levels = 1
        octree = createOctree(num_levels, pulse_mesh)
        fillOctreeZMatricesSoundSoft!(pulse_mesh, octree, wavenumber, distance_to_edge_tol, near_singular_tol, compression_distance, ACA_tol)
        @test isapprox(octree.nodes[1].node2node_Z_matrices[1], z_matrix,rtol = 1e-14)

        num_levels = 2
        octree = createOctree(num_levels, pulse_mesh)
        fillOctreeZMatricesSoundSoft!(pulse_mesh, octree, wavenumber, distance_to_edge_tol, near_singular_tol, compression_distance, ACA_tol)
        @test isapprox(octree.nodes[2].node2node_Z_matrices[1], z_matrix[[1,2],[1,2]], rtol = 1e-14)
        @test isapprox(octree.nodes[2].node2node_Z_matrices[2], z_matrix[[1,2],[3,4]], rtol = 1e-14)
        @test isapprox(octree.nodes[2].node2node_Z_matrices[4], z_matrix[[1,2],[7,8]], rtol = 1e-14)
        @test isapprox(octree.nodes[3].node2node_Z_matrices[1], z_matrix[[3,4],[1,2]], rtol = 1e-14)
        @test isapprox(octree.nodes[4].node2node_Z_matrices[4], z_matrix[[5,6],[7,8]], rtol = 1e-14)
        @test isapprox(octree.nodes[5].node2node_Z_matrices[3], z_matrix[[7,8],[5,6]], rtol = 1e-14)

        num_levels = 3
        large_compression_dist = 1e10 # all interactions use direct Z calculation
        octree = createOctree(num_levels, pulse_mesh)
        fillOctreeZMatricesSoundSoft!(pulse_mesh, octree, wavenumber, distance_to_edge_tol, near_singular_tol, large_compression_dist, ACA_tol)
        @test isapprox(octree.nodes[11].node2node_Z_matrices[5][1,1], z_matrix[5,6], rtol = 1e-14)
        @test isapprox(octree.nodes[11].node2node_Z_matrices[8][1,1], z_matrix[5,8], rtol = 1e-14)

        num_levels = 3
        octree = createOctree(num_levels, pulse_mesh)
        fillOctreeZMatricesSoundSoft!(pulse_mesh, octree, wavenumber, distance_to_edge_tol, near_singular_tol, compression_distance, ACA_tol)
        # first test types of elements of node2node_Z_matrices are correct
        UV_type = Tuple{Array{ComplexF64,2},Array{ComplexF64,2}}
        Z_type = Array{ComplexF64,2}
        @test typeof(octree.nodes[10].node2node_Z_matrices[1]) == UV_type
        @test typeof(octree.nodes[10].node2node_Z_matrices[2]) == Z_type
        @test typeof(octree.nodes[10].node2node_Z_matrices[3]) == UV_type
        @test typeof(octree.nodes[10].node2node_Z_matrices[4]) == Z_type
        @test typeof(octree.nodes[10].node2node_Z_matrices[5]) == Z_type
        @test typeof(octree.nodes[10].node2node_Z_matrices[6]) == Z_type
        @test typeof(octree.nodes[10].node2node_Z_matrices[7]) == Z_type
        @test typeof(octree.nodes[10].node2node_Z_matrices[8]) == UV_type
        # Now test a few exact values
        test_node = octree.nodes[11]
        src_node = octree.nodes[13]
        @test isapprox(test_node.node2node_Z_matrices[5][1,1], z_matrix[5,6], rtol = 1e-14)
        computeMatrixEntry(test_idx,src_idx) = computeZEntrySoundSoft(pulse_mesh, test_node, src_node, wavenumber, distance_to_edge_tol, near_singular_tol, test_idx, src_idx)
        sol_U, sol_V = computeMatrixACA(computeMatrixEntry, approximation_tol, length(test_node.element_idxs), length(src_node.element_idxs))
        @test isapprox(test_node.node2node_Z_matrices[8][1], sol_U, rtol = 1e-14)
        @test isapprox(test_node.node2node_Z_matrices[8][2], sol_V, rtol = 1e-14)
    end
    @testset "initializeOctree tests" begin
        num_levels = 1
        ele_centroids = [[1.0,1,0],[-1.0,-1,0]]
        node_centroid =[0.0,0.0,0.0]
        box_bounds = [-1.05*sqrt(2),1.05*sqrt(2)] # same for all dimensions
        sol_top_node_idx = 1
        sol_num_nodes = 1
        buffer = 0.1 # edge lengths
        test_octree = initializeOctree(num_levels, buffer, ele_centroids)
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
        buffer = 1
        ele_centroids = [[-1.0,0.2,0.0],[0.0,1.5,0.3],[0.0,0.0,1.0]]
        node_centroid = [-1,1.7,1.3] ./ 3
        max_ele_distance = 1.0 * (1+buffer/2)
        sol_x_bounds = [-1/3-max_ele_distance,-1/3+max_ele_distance]
        sol_top_node_idx = 1
        sol_num_nodes = 1
        test_octree = initializeOctree(num_levels, buffer, ele_centroids)
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
