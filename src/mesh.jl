using Parameters

include("../packages/gmsh.jl")

@with_kw struct PulseMesh
    num_elements::Int64
    nodes::Array{Float64, 2}
    elements::Array{Int64, 2}
    src_quadrature_points::Array{Float64, 3}
    src_quadrature_weights::Array{Float64, 1}
    test_quadrature_points::Array{Float64, 3}
    test_quadrature_weights::Array{Float64, 1}
end

function calculateQuadraturePoints(nodes::AbstractArray{Float64, 2}, elements::AbstractArray{Int64, 2}, area_quadrature_points::AbstractArray{Float64, 2})
    num_elements = size(elements)[1]
    num_points = size(area_quadrature_points)[2]
    quadrature_points = Array{Float64, 3}(undef, num_elements, num_points, 3)
    for element_idx in 1:num_elements
        ele_nodes = nodes[elements[element_idx,:], :]
        quadrature_points[element_idx, 1, :] = barycentric2Cartesian(ele_nodes, area_quadrature_points[:, 1])
    end
    quadrature_points
    # return Array{Float64, 3}(undef, size(elements)[1], size(area_quadrature_points)[2], 3)
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

function buildPulseMesh(mesh_filename::String, src_quadrature_rule::Array{Float64, 2}, test_quadrature_rule::Array{Float64, 2})
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

    PulseMesh(num_elements, nodes, elements, zeros((1,1,1)), zeros(1), zeros((1,1,1)), zeros(1))
end

function barycentric2Cartesian(nodes::Array{Float64, 2}, barycentric_coords::AbstractArray{Float64, 1})
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
