using LinearAlgebra
using Parameters

@with_kw mutable struct Octree
    num_levels::Int64 = 0
    # box_to_element_map::AbstractArray{Array{Int64,1},1} = [[]]
    box_centroids::AbstractArray{Array{Array{Float64,1},1},1} = [[[]]]
    box_bounds
end

function initializeOctree(num_levels, ele_centroids::AbstractArray{Array{Float64,1},1})
    num_boxes = 8^(num_levels-1)
    num_elements = length(ele_centroids)
    lvl0_box_center = sum(ele_centroids)/num_elements
    box_centroids = [[[0.0,0,0]]]
    #find edge length of level 0 box
    max_distance = 0.0
    for ele_idx = 1:num_elements
        distance = norm(lvl0_box_center - ele_centroids[ele_idx])
        if distance > max_distance
            max_distance = distance
        end
    end
    lvl0_edge_length = 2*max_distance
# left off computing and storing box bounds
    return(Octree(num_levels, box_centroids))
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
end
