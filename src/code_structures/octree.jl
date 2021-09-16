# dependencies: mesh.jl

################################################################################
# This file contains structs for and Octree and a Node and functions that      #
# create these structures.  Functions to fill node2node_X_matrices are in      #
# ACA_fill.jl.                                                                 #
################################################################################

using LinearAlgebra
using Parameters

mutable struct Node
    octree_level::Int64
    parent_idx::Int64
    children_idxs::Array{Int64,1}
    element_idxs::Array{Int64,1}
    bounds::Array{Array{Float64,1},1}
    centroid::Array{Float64,1}
    node2node_Z_matrices::Array{Any,1}
    node2node_dZdk_matrices::Array{Any,1}
end

@with_kw mutable struct Octree
    num_levels::Int64 = 0
    top_node_idx::Int64 = 0
    leaf_node_idxs::Array{Int64,1} = []
    nodes::Array{Node,1} = []
end

@views function computeNodeBounds(half_edge_length, node_centroid::Array{Float64,1})
    # Computes the boundaries of the node given half the length of a node edge
    # (assumes node is a cube) and the centroid of the node. Returns array of
    # arrays: [[lower_x,upper_x],[lower_y,upper_y],[lower_z,upper_z]]
    return([[node_centroid[1] - half_edge_length, node_centroid[1] + half_edge_length],
            [node_centroid[2] - half_edge_length, node_centroid[2] + half_edge_length],
            [node_centroid[3] - half_edge_length, node_centroid[3] + half_edge_length]])
end

@views function createChildren(parent_idx::Int64, parent_node::Node, ele_centroids::AbstractArray{Array{Float64,1},1})
    # creates the child nodes of parent_node only storing the ones containing
    # element centroids and returning all child nodes in the array children_nodes
    num_children = 8
    children_nodes = []
    child_level = parent_node.octree_level + 1
    child_edge_length = (parent_node.bounds[1][2] - parent_node.bounds[1][1])/2
    parent_ele_centroids = ele_centroids[parent_node.element_idxs]
    child_idx = 1
    for z_idx = 1:2 # the x, y and z idxs implicitly loop through all children
        z_bounds = [parent_node.centroid[3]-(2-z_idx)*child_edge_length,
                    parent_node.centroid[3]-(1-z_idx)*child_edge_length]
        for y_idx = 1:2
            y_bounds = [parent_node.centroid[2]-(2-y_idx)*child_edge_length,
                        parent_node.centroid[2]-(1-y_idx)*child_edge_length]
            for x_idx = 1:2
                x_bounds = [parent_node.centroid[1]-(2-x_idx)*child_edge_length,
                            parent_node.centroid[1]-(1-x_idx)*child_edge_length]
                child_bounds = [x_bounds, y_bounds, z_bounds]
                child_centroid = sum.(child_bounds) ./ 2
                child_element_idxs = []
                for local_ele_idx = 1:length(parent_ele_centroids)
                    ele_centroid = parent_ele_centroids[local_ele_idx]
                    # note: in below criteria, note that if no buffer is used in initializing the octree,
                    # then if a centroid lies on uppermost edges of octree domain, it doesn't get included in a node
                    if (((child_bounds[1][1] <= ele_centroid[1]) && (ele_centroid[1] < child_bounds[1][2])) &&
                        ((child_bounds[2][1] <= ele_centroid[2]) && (ele_centroid[2] < child_bounds[2][2])) &&
                        ((child_bounds[3][1] <= ele_centroid[3]) && (ele_centroid[3] < child_bounds[3][2])))
                        global_ele_idx = parent_node.element_idxs[local_ele_idx]
                        push!(child_element_idxs, global_ele_idx)
                    end
                end
                if isempty(child_element_idxs) == false
                    no_children_idxs, no_Z_matrices, no_dZdk_matrices = [], [], []
                    child_node = Node(child_level, parent_idx, no_children_idxs, child_element_idxs, child_bounds, child_centroid, no_Z_matrices, no_dZdk_matrices)
                    push!(children_nodes, child_node)
                end
                child_idx += 1
            end
        end
    end
    return(children_nodes)
end # function createChildren

function createOctree(num_levels::Int64, pulse_mesh::PulseMesh)
    # highest-level function that handles all octree construction
    # num_levels dictates the amount of levels in the octree
    # returns the octree with everything filled except node2node_Z_matrices in the nodes
    # note: this function has type inference issues, but is only called once so I think it is okay as long as issues arent propagating
    # note 2: It appears that the infernece issues disappear when called in higher level functions
    @unpack num_elements,
            elements,
            nodes = pulse_mesh
    parent_idx = 1
    buffer = 1e-4
    ele_centroids = Array{Array{Float64,1},1}(undef, num_elements)
    for ele_idx = 1:num_elements
        ele_centroids[ele_idx] = computeCentroid(nodes[elements[ele_idx,:],:]::Array{Float64,2}) # declaring type avoids inference issue
    end
    octree = initializeOctree(num_levels, buffer, ele_centroids)
    if num_levels > 1
        fillOctreeNodes!(parent_idx, octree, ele_centroids)
    else
        octree.leaf_node_idxs = [1]
    end
    return(octree)
end # function createOctree

@views function fillOctreeNodes!(parent_idx::Int64, octree::Octree, ele_centroids::AbstractArray{Array{Float64,1},1})
    # recursive function that spawns all child nodes down to the leaf level
    # starting from the node given by the global index parent_idx
    # returns nothing, updates octree instance with new nodes and information
    parent_level = octree.nodes[parent_idx].octree_level
    current_num_nodes = length(octree.nodes)
    child_nodes = createChildren(parent_idx, octree.nodes[parent_idx], ele_centroids)
    append!(octree.nodes, child_nodes)
    octree.nodes[parent_idx].children_idxs = [i+current_num_nodes for i=1:length(child_nodes)]
    if parent_level < (octree.num_levels - 1)
        for local_child_idx = 1:length(child_nodes)
            global_child_idx = local_child_idx + current_num_nodes
            fillOctreeNodes!(global_child_idx, octree, ele_centroids)
        end
    else # at leaf level; stop recursion
        append!(octree.leaf_node_idxs, octree.nodes[parent_idx].children_idxs)
    end
end # function fillOctreeNodes!

@views function initializeOctree(num_levels::Int64, buffer, ele_centroids::AbstractArray{Array{Float64,1},1})
    # Creates in octree instance with only the level 1 node encapsulating all elements
    # buffer increases node edge length by fraction of its required edge length to make sure all elements are included
    level1 = 1
    no_parent_idx = 0 # the zero indicates it is at the highest level
    no_children_idxs = [] # empty array indicates it is a leaf
    no_Z_matrices = []
    no_leaves = []
    num_elements = length(ele_centroids)
    all_ele_idxs = [i for i=1:num_elements]
    level1_centroid = sum(ele_centroids)/num_elements
    max_distance = 0.0 # from centroid of node
    for ele_idx = 1:num_elements
        distance = norm(level1_centroid - ele_centroids[ele_idx])
        if distance > max_distance
            max_distance = distance
        end
    end
    level1_half_edge_length = (1+buffer/2)*max_distance
    level1_bounds = computeNodeBounds(level1_half_edge_length, level1_centroid)
    level1_node = Node(level1, no_parent_idx, no_children_idxs, all_ele_idxs, level1_bounds, level1_centroid, no_Z_matrices, no_Z_matrices)
    level1_node_idx = 1
    return(Octree(num_levels, level1_node_idx, no_leaves, [level1_node]))
end # function initializeOctree
