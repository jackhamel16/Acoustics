# dependencies: mesh.jl quadrature.jl

using LinearAlgebra

function rhsFill(pulse_mesh::PulseMesh,
                 fieldFunc::Function,
                 rhs::AbstractArray{ComplexF64, 1},
                 normal_derivative=false)
    @unpack num_elements,
            elements,
            areas,
            nodes,
            normals,
            test_quadrature_points,
            test_quadrature_weights = pulse_mesh
    if normal_derivative == true # need to redefine function passed to quadrature with normal
        for element_idx in 1:num_elements
            triangle_nodes = getTriangleNodes(element_idx, elements, nodes)
            triangle_area = areas[element_idx]
            fieldFuncNormal(x,y,z) = fieldFunc(x,y,z,normals[element_idx,:])
            rhs[element_idx] += -1 * gaussQuadrature(triangle_area,
                                                     fieldFuncNormal,
                                                     test_quadrature_points[element_idx],
                                                     test_quadrature_weights)
        end
    else
        for element_idx in 1:num_elements
            triangle_nodes = getTriangleNodes(element_idx, elements, nodes)
            triangle_area = areas[element_idx]
            rhs[element_idx] += -1 * gaussQuadrature(triangle_area,
                                                     fieldFunc,
                                                     test_quadrature_points[element_idx],
                                                     test_quadrature_weights)
        end
    end
end

function matrixFill(pulse_mesh::PulseMesh,
                    testIntegrand::Function,
                    z_matrix::AbstractArray{ComplexF64, 2})
    @unpack num_elements,
            elements,
            areas,
            nodes,
            test_quadrature_points,
            test_quadrature_weights = pulse_mesh
    for src_idx in 1:num_elements
        for test_idx in 1:num_elements
            is_singular = (src_idx == test_idx)
            test_nodes = getTriangleNodes(test_idx, elements, nodes)
            src_nodes = getTriangleNodes(src_idx, elements, nodes)
            testIntegrandXYZ(x,y,z) = testIntegrand([x,y,z], src_idx, is_singular)
            z_matrix[test_idx, src_idx] += gaussQuadrature(areas[test_idx],
                                                           testIntegrandXYZ,
                                                           test_quadrature_points[test_idx],
                                                           test_quadrature_weights)
        end
    end
end

function nodeMatrixFill!(pulse_mesh::PulseMesh,
                        test_node::Node,
                        src_node::Node,
                        testIntegrand::Function,
                        sub_z_matrix::AbstractArray{ComplexF64, 2})
    @unpack elements,
            areas,
            nodes,
            test_quadrature_points,
            test_quadrature_weights = pulse_mesh
    num_src_elements = length(src_node.element_idxs)
    num_test_elements = length(test_node.element_idxs)
    for local_src_idx in 1:num_src_elements
        global_src_idx = src_node.element_idxs[local_src_idx]
        for local_test_idx in 1:num_test_elements
            global_test_idx = test_node.element_idxs[local_test_idx]
            is_singular = (global_src_idx == global_test_idx)
            test_tri_nodes = getTriangleNodes(global_test_idx, elements, nodes)
            src_tri_nodes = getTriangleNodes(global_src_idx, elements, nodes)
            testIntegrandXYZ(x,y,z) = testIntegrand([x,y,z], global_src_idx, is_singular)
            sub_z_matrix[local_test_idx, local_src_idx] += gaussQuadrature(areas[global_test_idx],
                                                           testIntegrandXYZ,
                                                           test_quadrature_points[global_test_idx],
                                                           test_quadrature_weights)
        end
    end
end
