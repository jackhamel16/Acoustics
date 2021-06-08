using LinearAlgebra
using Parameters

mutable struct Node
    octree_level::Int64
    parent_idx::Int64
    children_idxs::Array{Int64,1}
    element_idxs::Array{Int64,1}
    bounds::Array{Array{Float64,1},1}
    centroid::Array{Float64,1}
end

@with_kw mutable struct Octree
    num_levels::Int64 = 0
    top_node_idx::Int64 = 0
    leaf_node_idxs::Array{Int64,1} = []
    nodes::Array{Node,1} = []
end

@views function computeNodeBounds(half_edge_length, node_centroid::Array{Float64,1})
    return([[node_centroid[1] - half_edge_length, node_centroid[1] + half_edge_length],
            [node_centroid[2] - half_edge_length, node_centroid[2] + half_edge_length],
            [node_centroid[3] - half_edge_length, node_centroid[3] + half_edge_length]])
end

@views function createChildren(parent_idx::Int64, parent_node::Node, ele_centroids::AbstractArray{Array{Float64,1},1})
    # creates the child nodes of parent_node only storing the ones containing
    # element centroids and returning each node in an array
    num_children = 8
    no_children_idxs = []
    children_nodes = []
    child_level = parent_node.octree_level + 1
    child_edge_length = (parent_node.bounds[1][2] - parent_node.bounds[1][1])/2
    parent_ele_centroids = ele_centroids[parent_node.element_idxs]
    child_idx = 1
    for z_idx = 1:2 # the x, y and z idxs implicitly loop through children
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
                    if (((child_bounds[1][1] <= ele_centroid[1]) && (ele_centroid[1] < child_bounds[1][2])) &&
                        ((child_bounds[2][1] <= ele_centroid[2]) && (ele_centroid[2] < child_bounds[2][2])) &&
                        ((child_bounds[3][1] <= ele_centroid[3]) && (ele_centroid[3] < child_bounds[3][2])))
                        global_ele_idx = parent_node.element_idxs[local_ele_idx]
                        push!(child_element_idxs, global_ele_idx)
                    end
                end
                if isempty(child_element_idxs) == false
                    child_node = Node(child_level, parent_idx, no_children_idxs, child_element_idxs, child_bounds, child_centroid)
                    push!(children_nodes, child_node)
                end
                child_idx += 1
            end
        end
    end
    return(children_nodes)
end # createChildren

@views function createOctree(num_levels::Int64, pulse_mesh::PulseMesh)
    # highest-level function that handles all octree construction
    parent_idx = 1
    buffer = 1e-4
    ele_centroids = Array{Array{Float64,1},1}(undef, pulse_mesh.num_elements)
    for ele_idx = 1:pulse_mesh.num_elements
        ele_centroids[ele_idx] = computeCentroid(pulse_mesh.nodes[pulse_mesh.elements[ele_idx,:],:])
    end
    octree = initializeOctree(num_levels, buffer, ele_centroids)
    if num_levels > 1
        fillOctreeNodes!(parent_idx, octree, ele_centroids)
    else
        octree.leaf_node_idxs = [1]
    end
    return(octree)
end

@views function fillOctreeNodes!(parent_idx::Int64, octree::Octree, ele_centroids::AbstractArray{Array{Float64,1},1})
    # recursive function that spawns all child nodes down to the leaf level
    # starting from the node given by the global index parent_idx
    # returns nothing, updates octree instance with new nodes and information
    parent_level = octree.nodes[parent_idx].octree_level
    current_num_nodes = length(octree.nodes)
    child_nodes = createChildren(parent_idx, octree.nodes[parent_idx], ele_centroids)
    append!(octree.nodes, child_nodes)
    octree.nodes[parent_idx].children_idxs = [i+current_num_nodes for i=1:length(child_nodes)]
    if (parent_level + 1) < octree.num_levels
        for local_child_idx = 1:length(child_nodes)
            global_child_idx = local_child_idx + current_num_nodes
            fillOctreeNodes!(global_child_idx, octree, ele_centroids)
        end
    end
    if parent_level == (octree.num_levels - 1)
        append!(octree.leaf_node_idxs, octree.nodes[parent_idx].children_idxs)
    end
end # fillOctreeNodes!

@views function initializeOctree(num_levels::Int64, buffer, ele_centroids::AbstractArray{Array{Float64,1},1})
    # Creates in octree instance with only the level 1 node, the node
    # encapsulating all elements, filled.
    # buffer increases node edge length by fraction of its required edge length to make sure all elements are included
    level1 = 1
    no_parent_idx = 0 # the zero indicates it is at the highest level
    no_children_idxs = [] # empty array indicates it is a leaf
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
    level1_node = Node(level1, no_parent_idx, no_children_idxs, all_ele_idxs, level1_bounds, level1_centroid)
    level1_node_idx = 1
    return(Octree(num_levels, level1_node_idx, no_leaves, [level1_node]))
end # initializeOctree
