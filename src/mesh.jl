include("../packages/gmsh.jl")
# using ..gmsh

struct PulseMesh
    num_elements::Int64
    num_coord_dims::Int64
    nodes_per_triangle::Int64
    nodes::Array{Float64, 2}
    elements::Array{Int64, 2}
    centroids::Array{Float64, 2}
end

function computeCentroid(vertices::Array{Float64,2})
    # Computes the centroid of a triangle with 3D coordinates
    # vertices is a 3x3 array. each row is a point and col a coordinate
    [(vertices[1,1]+vertices[2,1]+vertices[3,1])/3,
     (vertices[1,2]+vertices[2,2]+vertices[3,2])/3,
     (vertices[1,3]+vertices[2,3]+vertices[3,3])/3]
end

function reshapeMeshArray(array::Array{T,1}, num_cols, type=T) where T<:Number
    # reshapes 1D array to 2D with each row corresponding to an elements
    # array is the Array to be reshaped
    # num_cols is number of columns to have in the reshaped array
    convert(Array{type}, transpose(reshape(array, (num_cols, Integer(length(array)/num_cols)))))
end

function buildPulseMesh(mesh_filename::String)
    # Builds a PulseMesh object based the mesh at mesh_filename
    num_coord_dims = 3
    nodes_per_triangle = 3

    gmsh.initialize();
    gmsh.open(mesh_filename)
    nodes = reshapeMeshArray(gmsh.model.mesh.getNodes()[2], num_coord_dims)
    elements_idx = size(gmsh.model.mesh.getElements()[3])[1] #if 2, first array are line elements, if 1 they are triangles
    elements = reshapeMeshArray(gmsh.model.mesh.getElements()[3][elements_idx], nodes_per_triangle)
    gmsh.finalize()

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
    PulseMesh(num_elements, num_coord_dims, nodes_per_triangle, nodes, elements, centroids)
end

function barycentric2Cartesian(nodes::Array{Float64, 2}, barycentric_coords::Array{Float64, 1})
    cartesian_coords = zeros(3)
    for node_idx in 1:3
        cartesian_coords += barycentric_coords[node_idx] * nodes[node_idx,:]
    end
    cartesian_coords
end
