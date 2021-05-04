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
            # z_matrix[test_idx, src_idx] += integrateTriangle(test_nodes,
            #                                                   testIntegrandXYZ,
            #                                                   test_quadrature_points[test_idx],
            #                                                   test_quadrature_weights)
        end
    end
end
