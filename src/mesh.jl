include("../packages/gmsh.jl")

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
    node_tags, node_xyzs = gmsh.model.mesh.getNodes(-1,-1)
    element_types, element_tags, element_nodes = gmsh.model.mesh.getElements(-1,-1)
    gmsh.finalize()
    triangles_idx = findall(x->x==2, element_types)[1]
    num_nodes = size(node_tags)[1]
    num_elements = size(element_tags[triangles_idx])[1]
    nodes = Array{Float64, 2}(undef, num_nodes, num_coord_dims)
    for node_idx in 1:num_nodes
        tag_idx = node_tags[node_idx]
        xyz_idx = (node_idx-1)*num_coord_dims + 1
        nodes[tag_idx, :] = node_xyzs[xyz_idx:xyz_idx+nodes_per_triangle-1]
    end
    elements = reshapeMeshArray(element_nodes[triangles_idx], num_coord_dims)

    centroids = Array{Float64, 2}(undef, num_elements, num_coord_dims)
    for element_idx in 1:num_elements
        element_vertices = nodes[elements[element_idx,:],:]
        centroids[element_idx,:] = computeCentroid(element_vertices)
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

function exportSourcesGmsh(mesh_filename::String,
                           output_filename::String,
                           sources::Array{Float64, 1})
    gmsh.initialize();
    gmsh.open(mesh_filename);
    models = gmsh.model.list();
    model_name = models[1];
    (element_types, element_tags, node_tags) = gmsh.model.mesh.getElements(-1,-1);
    triangles_idx = findall(x->x==2, element_types)[1]
    TriangleTags = element_tags[triangles_idx];

    gmsh.write(string(output_filename,".pos"));  # export mesh

    step_identifier = 1
    time_identifier = 1
    num_components = 1
    view_name = "mesh scalar sources"

    tag_view = gmsh.view.add(view_name);
    gmsh.view.addHomogeneousModelData(tag_view, step_identifier, model_name,
                                      "ElementData", TriangleTags, sources,
                                      time_identifier, num_components)

    write, append = false, true
    gmsh.view.write(tag_view, string(output_filename,".pos"), write);
end
