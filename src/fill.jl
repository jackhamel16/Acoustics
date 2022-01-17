# dependencies: mesh.jl quadrature.jl octree.jl

################################################################################
# This file contains the functions that fill the Z matrix and V (rhs) vectors  #
# to be used in non-ACA solve functions for any of the integral equations.     #
################################################################################

using LinearAlgebra

function rhsFill!(pulse_mesh::PulseMesh,
                 fieldFunc::Function,
                 rhs::AbstractArray{ComplexF64, 1})
    # Used in solving regular IEs to fill RHS vector
    @unpack num_elements,
            elements,
            areas,
            nodes,
            normals,
            test_quadrature_points,
            test_quadrature_weights = pulse_mesh
    Threads.@threads for element_idx in 1:num_elements
        triangle_nodes = getTriangleNodes(element_idx, elements, nodes)
        triangle_area = areas[element_idx]
        rhs[element_idx] += -1 * gaussQuadrature(triangle_area,
                                                 fieldFunc,
                                                 test_quadrature_points[element_idx],
                                                 test_quadrature_weights)
    end
end # function rhsFill!

function rhsNormalDerivFill!(pulse_mesh::PulseMesh,
                 fieldFunc::Function,
                 rhs::AbstractArray{ComplexF64, 1})
    # Used in solving IE normal derivatives to fill RHS vector. The only difference
    #   is the normal vector of the test element is passed to fieldFunc in addition to x,y,z
    @unpack num_elements,
            elements,
            areas,
            nodes,
            normals,
            test_quadrature_points,
            test_quadrature_weights = pulse_mesh
    for element_idx in 1:num_elements
        triangle_nodes = getTriangleNodes(element_idx, elements, nodes)
        triangle_area = areas[element_idx]
        fieldFuncWithNormal(x,y,z) = fieldFunc(x,y,z,normals[element_idx,:])
        rhs[element_idx] += -1 * gaussQuadrature(triangle_area,
                                                 fieldFuncWithNormal,
                                                 test_quadrature_points[element_idx],
                                                 test_quadrature_weights)
    end
end # function rhsNormalDerivFill!

function matrixFill!(pulse_mesh::PulseMesh,
                    testIntegrand::Function,
                    z_matrix::AbstractArray{ComplexF64, 2})
    # Used in solving regular IEs to fill Z matrix
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
end # function matrixFill!

function matrixNormalDerivFill!(pulse_mesh::PulseMesh,
                    testIntegrand::Function,
                    z_matrix::AbstractArray{ComplexF64, 2})
    # Used in solving IE normal derivatives to fill Z matrix. The only difference
    #   is the normal vector of the test element is also passed to testIntegrand
    @unpack num_elements,
            elements,
            areas,
            normals,
            nodes,
            test_quadrature_points,
            test_quadrature_weights = pulse_mesh
    for src_idx in 1:num_elements
        for test_idx in 1:num_elements
            is_singular = (src_idx == test_idx)
            test_nodes = getTriangleNodes(test_idx, elements, nodes)
            src_nodes = getTriangleNodes(src_idx, elements, nodes)
            testIntegrandXYZ(x,y,z) = testIntegrand([x,y,z], src_idx, normals[test_idx,:], is_singular)
            z_matrix[test_idx, src_idx] += gaussQuadrature(areas[test_idx],
                                                           testIntegrandXYZ,
                                                           test_quadrature_points[test_idx],
                                                           test_quadrature_weights)
        end
    end
end # function matrixNormalDerivFill!
