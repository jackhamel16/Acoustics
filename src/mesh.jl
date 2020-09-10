include("../packages/gmsh.jl")
# using ..gmsh

struct PulseMesh
    num_coord_dims::Integer
    nodes_per_triangle::Integer
    nodes::Array{Float64, 2}
    elements::Array{Integer, 2}
    centroids::Array{Float64, 2}
end

function computeCentroid(vertices::Array{Float64,2})
    # Computes the centroid of a triangle with 3D coordinates
    # vertices is a 3x3 array. each row is a point and col a coordinate
    [(vertices[1,1]+vertices[2,1]+vertices[3,1])/3,
     (vertices[1,2]+vertices[2,2]+vertices[3,2])/3,
     (vertices[1,3]+vertices[2,3]+vertices[3,3])/3]
end

function reshapeMeshArray(array::Array{T,1}, num_cols) where T<:Number
    # reshapes 1D array to 2D with each row corresponding to an elements
    # array is the Array to be reshaped
    # num_cols is number of columns to have in the reshaped array
    convert(Array{T}, transpose(reshape(array, (num_cols, Integer(length(array)/num_cols)))))
end

function buildPulseMesh(mesh_filename::String)
    # Builds a PulseMesh object based the mesh at mesh_filename
    num_coord_dims = 3
    nodes_per_triangle = 3

    gmsh.initialize();
    gmsh.open(mesh_filename)
    nodes = reshapeMeshArray(gmsh.model.mesh.getNodes()[2], num_coord_dims)
    elements = reshapeMeshArray(gmsh.model.mesh.getElements()[3][2], nodes_per_triangle)

    num_elements = size(elements)[1]
    centroids = Array{Float64, 2}(undef, num_elements, num_coord_dims)
    for element_idx in 1:num_elements
        vertices = Array{Float64,2}(undef, 3, 3)
        for loop_node_idx in 1:nodes_per_triangle
            local_node_idx = elements[element_idx, loop_node_idx]
            node_coords = nodes[local_node_idx, :]
            vertices[loop_node_idx,:] = node_coords
        end
        centroids[element_idx,:] = computeCentroid(vertices)
    end

    PulseMesh(num_coord_dims, nodes_per_triangle, nodes, elements, centroids)
end

# function barycentric2Cartesian()
#     0
# end
