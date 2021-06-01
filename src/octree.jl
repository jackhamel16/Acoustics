using LinearAlgebra
using Parameters

# @with_kw mutable struct Octree
#     num_levels::Int64 = 0
#     box_to_element_map::AbstractArray{Array{Array{Int64,1},1},1} = [[[]]]
#     box_centroids::AbstractArray{Array{Array{Float64,1},1},1} = [[[]]]
#     box_bounds::AbstractArray{Array{Array{Tuple{Float64,Float64},1},1},1} = [[[()]]]
# end

mutable struct Node
    parent_idx::Int64
    children_idxs::Array{Int64,1}
    element_idxs::Array{Int64,1}
    bounds::Array{Array{Float64,1},1}
    centroid::Array{Float64,1}
end

mutable struct Octree
    num_levels::Int64
    top_node_idx::Int64
    nodes::Array{Node,1}
end

function computeNodeBounds(half_edge_length, node_centroid::Array{Float64,1})
    return([[node_centroid[1] - half_edge_length, node_centroid[1] + half_edge_length],
            [node_centroid[2] - half_edge_length, node_centroid[2] + half_edge_length],
            [node_centroid[3] - half_edge_length, node_centroid[3] + half_edge_length]])
end

function initializeOctree(num_levels::Int64, ele_centroids::AbstractArray{Array{Float64,1},1})
    # Creates in octree instance with only the level 1 node, the node
    # encapsulating all elements, filled.
    no_parent_idx = 0 # the zero indicates it is at the highest level
    no_children_idxs = [] # empty array indicates it is a leaf
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
    level1_edge_length = 2*max_distance
    level1_bounds = computeNodeBounds(max_distance, level1_centroid)
    level1_node = Node(no_parent_idx, no_children_idxs, all_ele_idxs, level1_bounds, level1_centroid)
    level1_node_idx = 1
    return(Octree(num_levels, level1_node_idx, [level1_node]))
end # initializeOctree

#     num_boxes = 8^(num_levels-1)
#     num_elements = length(ele_centroids)
#     lvl0_box_center = sum(ele_centroids)/num_elements
#     box_centroids = [[[0.0,0,0]]]
#     #find edge length of level 0 box
#     max_distance = 0.0
#     for ele_idx = 1:num_elements
#         distance = norm(lvl0_box_center - ele_centroids[ele_idx])
#         if distance > max_distance
#             max_distance = distance
#         end
#     end
#     lvl0_edge_length = 2*max_distance
# # left off computing and storing box bounds
#     lvl0_x_centroid = box_centroids[1][1][1]
#     lvl0_y_centroid = box_centroids[1][1][2]
#     lvl0_z_centroid = box_centroids[1][1][3]
#     lvl0_x_bounds = (lvl0_x_centroid - max_distance, lvl0_x_centroid + max_distance)
#     lvl0_y_bounds = (lvl0_y_centroid - max_distance, lvl0_y_centroid + max_distance)
#     lvl0_z_bounds = (lvl0_z_centroid - max_distance, lvl0_z_centroid + max_distance)
#     lvl0_bounds = [[lvl0_x_bounds, lvl0_y_bounds, lvl0_z_bounds]]
#     box_bounds = [lvl0_bounds]
#     return(Octree(num_levels=num_levels, box_centroids=box_centroids, box_bounds=box_bounds))
    # box_bounds = Array{Array{Float64,1},1}(undef, num_boxes)

    # octree = Octree()
    # box_to_element_map = Array{Array{Int64,1},1}(undef, num_boxes)
    # box_centroids = [[0,0,0]]
    # for box_idx = 1:num_boxes
    #     box_bounds[box_idx] = []
    #     box_to_element_map[box_idx] = []
    #     for ele_idx = 1:num_elements
    #
    #         # push!(box_to_element_map[1], ele_idx)
    #     end
    # end
    # return(Octree(box_to_element_map, box_centroids))
