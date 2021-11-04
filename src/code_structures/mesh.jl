using LinearAlgebra
using Parameters

include("../../packages/gmsh.jl")

@with_kw mutable struct PulseMesh
    num_elements::Int64 = 0
    nodes::Array{Float64, 2} = Array{Float64, 2}(undef, 0, 0)
    elements::Array{Int64, 2} = Array{Int64, 2}(undef, 0, 0)
    areas::Array{Float64, 1} = Array{Float64, 1}(undef, 0)
    normals::Array{Float64, 2} = Array{Float64, 2}(undef, 0, 0)
    src_quadrature_rule::Array{Float64, 2} = Array{Float64, 2}(undef, 0, 0)
    test_quadrature_rule::Array{Float64, 2} = Array{Float64, 2}(undef, 0, 0)
    src_quadrature_points::Array{Array{Float64, 2},1} = Array{Array{Float64, 2},1}(undef, 0)
    src_quadrature_weights::Array{Float64, 1} = Array{Float64, 1}(undef, 0)
    test_quadrature_points::Array{Array{Float64, 2},1} = Array{Array{Float64, 2},1}(undef, 0)
    test_quadrature_weights::Array{Float64, 1} = Array{Float64, 1}(undef, 0)
    Z_factors::LU{T,Array{T,2}} where T = lu(ones(1,1))
    RHS::Array{T,1} where T = Array{ComplexF64, 1}(undef, 0)
end

@views function calculateQuadraturePoints(nodes::AbstractArray{Float64, 2},
                                          elements::AbstractArray{Int64, 2},
                                          area_quadrature_points::AbstractArray{Float64, 2})
    # Computes the quadrature points on a triangle described by elements and nodes
    # from the triangle area coordinates of the quadrature points
    # nodes in an array of global nodes
    # array of node labels of all elements
    num_elements = size(elements)[1]
    num_points = size(area_quadrature_points)[2]
    quadrature_points = Array{Array{Float64, 2}}(undef, num_elements)
    quadrature_points_one_element = Array{Float64, 2}(undef, 3, num_points)
    for element_idx in 1:num_elements
        ele_nodes = getTriangleNodes(element_idx, elements, nodes)
        for pnt_idx in 1:num_points
            quadrature_points_one_element[:, pnt_idx] = barycentric2Cartesian(ele_nodes,
                                                                    area_quadrature_points[:, pnt_idx])
        end
        quadrature_points[element_idx] = copy(quadrature_points_one_element)
    end
    quadrature_points
end

function computeCentroid(vertices::AbstractArray{T,2}) where T
    # Computes the centroid of a triangle with 3D coordinates
    # vertices is a 3x3 array. each row is a point and col a coordinate
    [(vertices[1,1]+vertices[2,1]+vertices[3,1])/3,
     (vertices[1,2]+vertices[2,2]+vertices[3,2])/3,
     (vertices[1,3]+vertices[2,3]+vertices[3,3])/3]
end

@views function getTriangleNodes(element_idx::Int64, elements::Array{Int64, 2}, nodes::Array{Float64, 2})
    # Gets nodes of triangle specified by element_idx
    # (does not have dedicated unit test right now)
    # This function is much faster than fancy indexing e.g nodes[elements[element_idx,:],:]
    triangle_nodes = Array{Float64, 2}(undef, 3, 3)
    for node_idx_local in 1:3
        node_idx_global = elements[element_idx, node_idx_local]
        triangle_nodes[node_idx_local,:] = nodes[node_idx_global,:]
    end
    triangle_nodes
end

@views function reshapeMeshArray(array::Array{T,1}, num_cols, type=T) where T<:Number
    # reshapes 1D array to 2D with each row corresponding to an elements
    # array is the Array to be reshaped
    # num_cols is number of columns to have in the reshaped array
    convert(Array{type}, transpose(reshape(array, (num_cols, Integer(length(array)/num_cols)))))
end

@views function buildPulseMesh(mesh_filename::String,
                               src_quadrature_rule::AbstractArray{Float64, 2},
                               test_quadrature_rule::AbstractArray{Float64, 2})
    # Builds a PulseMesh object based the mesh at mesh_filename
    # leaves Z_factors and RHS as defaults to be filled in other WS or solve routines
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
    elements = reshapeMeshArray(element_nodes[triangles_idx], num_coord_dims, Int64)

    areas = Array{Float64, 1}(undef, num_elements)
    normals = Array{Float64, 2}(undef, num_elements, 3)
    for ele_idx in 1:num_elements
        ele_nodes = getTriangleNodes(ele_idx, elements, nodes)
        normal_non_unit = cross(ele_nodes[2,:]-ele_nodes[1,:], ele_nodes[3,:]-ele_nodes[1,:])
        area2 = norm(normal_non_unit)
        areas[ele_idx] = 0.5 * area2
        normals[ele_idx,:] = normal_non_unit ./ area2
    end

    PulseMesh(num_elements,
              nodes,
              elements,
              areas,
              normals,
              src_quadrature_rule,
              test_quadrature_rule,
              calculateQuadraturePoints(nodes, elements, src_quadrature_rule[1:3,:]),
              src_quadrature_rule[4, :],
              calculateQuadraturePoints(nodes, elements, test_quadrature_rule[1:3,:]),
              test_quadrature_rule[4, :],
              lu(ones(1,1)),
              Array{ComplexF64, 1}(undef, 0))
end

@views function barycentric2Cartesian(nodes::AbstractArray{Float64, 2},
                                      barycentric_coords::AbstractArray{Float64, 1})
    cartesian_coords = zeros(3)
    for node_idx in 1:3
        cartesian_coords += barycentric_coords[node_idx] * nodes[node_idx,:]
    end
    cartesian_coords
end

@views function calculateTriangleArea(nodes::AbstractArray{Float64, 2})
    0.5*norm(cross(nodes[2,:]-nodes[1,:], nodes[3,:]-nodes[1,:]))
end

function exportSourcesBundled(mesh_filename::String, tag::String, sources::AbstractArray{T, 1}) where T <: Number
    # bundled exportSourcesGmsh calls to clean up below
    exportSourcesGmsh(mesh_filename, string("sources_real",tag), real.(sources))
    exportSourcesGmsh(mesh_filename, string("sources_imag",tag), imag.(sources))
    exportSourcesGmsh(mesh_filename, string("sources_mag",tag), abs.(sources))
end

function exportSourcesGmsh(mesh_filename::String,
                           output_filename::String,
                           sources::AbstractArray{Float64, 1})
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
