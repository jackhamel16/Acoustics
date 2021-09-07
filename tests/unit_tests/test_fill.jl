using Test
using LinearAlgebra

include("../../src/excitation.jl")
include("../../src/code_structures/mesh.jl")
include("../../src/quadrature.jl")
include("../../src/code_structures/octree.jl")

include("../../src/fill.jl")

include("../../src/greens_functions.jl")

@testset "fill tests" begin
    @testset "rhsFill! tests" begin
        planeWave(wavevector, r_test) = exp(-im*dot(wavevector, r_test))

        wavevector = [0.0, 0.0, 1/10]
        r_test = [1/3, 1/3, 1.5]
        nodes_global = [0.0 0.0 1.5; 1.0 0.0 1.5; 0.0 1.0 1.5]
        area = 0.5
        elements = [1 2 3]
        num_elements = 1
        fieldFunc(x,y,z) = planeWave(wavevector, [x,y,z])
        # simple 1 integration point test 1 triangle
        rhs_solution = -1 * area * planeWave(wavevector, r_test)
        pulse_mesh1 = PulseMesh(num_elements=num_elements, nodes=nodes_global, elements=elements,
                               areas=[area], test_quadrature_rule=gauss1rule,
                               test_quadrature_points=calculateQuadraturePoints(nodes_global, elements, gauss1rule[1:3,:]),
                               test_quadrature_weights=gauss1rule[4,:])
        rhs = zeros(ComplexF64, num_elements)
        rhsFill!(pulse_mesh1, fieldFunc, rhs)
        @test isapprox(rhs[1], rhs_solution, rtol=1e-15)
        # 7 point integration test 1 triangle
        pulse_mesh7 = PulseMesh(num_elements=num_elements, nodes=nodes_global, elements=elements,
                               areas=[area], test_quadrature_rule=gauss7rule,
                               test_quadrature_points=calculateQuadraturePoints(nodes_global, elements, gauss7rule[1:3,:]),
                               test_quadrature_weights=gauss7rule[4,:])
        rhs_solution = -1 * gaussQuadrature(area, fieldFunc, pulse_mesh7.test_quadrature_points[1], pulse_mesh7.test_quadrature_weights)
        rhs = zeros(ComplexF64, num_elements)
        rhsFill!(pulse_mesh7, fieldFunc, rhs)
        @test isapprox(rhs[1], rhs_solution, rtol=1e-15)

        # Next three tests involve two triangles
        wavevector = [0.0, 0.0, 1/10]
        nodes_global = [0.0 0.0 1.5; 1.0 0.0 1.5; 1.0 1.0 1.5; 0.0 1.0 1.5]
        areas = [0.5, 0.5]
        elements = [1 2 4; 2 3 4]
        num_elements = 2
        fieldFunc(x,y,z) = planeWave(wavevector, [x,y,z])
        centroids = [1/3 1/3 1.5; 2/3 2/3 1.5]
        rhs_solution = -1 .* [areas[1]*planeWave(wavevector, centroids[1,:]), areas[2]*planeWave(wavevector, centroids[2,:])]
        pulse_mesh1 = PulseMesh(num_elements=num_elements, nodes=nodes_global, elements=elements,
                               areas=areas, test_quadrature_rule=gauss1rule,
                               test_quadrature_points=calculateQuadraturePoints(nodes_global, elements, gauss1rule[1:3,:]),
                               test_quadrature_weights=gauss1rule[4,:])
        rhs = zeros(ComplexF64, num_elements)
        rhsFill!(pulse_mesh1, fieldFunc, rhs)
        @test isapprox(rhs, rhs_solution, rtol=1e-15)
        #
        wavevector = [0.0, 0.0, 1/10]
        nodes_global = [0.0 0.0 1.5; 1.0 0.0 1.5; 1.0 1.0 1.0; 0.0 1.0 1.5]
        areas = [0.5, 0.6123724356957945]
        elements = [1 2 4; 2 3 4]
        num_elements = 2
        fieldFunc(x,y,z) = planeWave(wavevector, [x,y,z])
        centroids = [1/3 1/3 1.5; 2/3 2/3 1+1/3]
        rhs_solution = -1 .* [areas[1]*planeWave(wavevector, centroids[1,:]), areas[2]*planeWave(wavevector, centroids[2,:])]
        pulse_mesh1 = PulseMesh(num_elements=num_elements, nodes=nodes_global, elements=elements,
                               areas=areas, test_quadrature_rule=gauss1rule,
                               test_quadrature_points=calculateQuadraturePoints(nodes_global, elements, gauss1rule[1:3,:]),
                               test_quadrature_weights=gauss1rule[4,:])
        rhs = zeros(ComplexF64, num_elements)
        rhsFill!(pulse_mesh1, fieldFunc, rhs)
        @test isapprox(rhs, rhs_solution, rtol=1e-15)

        wavevector = [1/30, 1/20, 1/15]
        nodes_global = [0.0 0.0 1.5; 1.0 0.0 1.5; 1.0 1.0 1.0; 0.0 1.0 1.5]
        areas = [0.5, 0.6123724356957945]
        elements = [1 2 4; 2 3 4]
        num_elements = 2
        fieldFunc(x,y,z) = planeWave(wavevector, [x,y,z])
        centroids = [1/3 1/3 1.5; 2/3 2/3 1+1/3]
        rhs_solution = -1 .* [areas[1]*planeWave(wavevector, centroids[1,:]), areas[2]*planeWave(wavevector, centroids[2,:])]
        pulse_mesh1 = PulseMesh(num_elements=num_elements, nodes=nodes_global, elements=elements,
                               areas=areas, test_quadrature_rule=gauss1rule,
                               test_quadrature_points=calculateQuadraturePoints(nodes_global, elements, gauss1rule[1:3,:]),
                               test_quadrature_weights=gauss1rule[4,:])
        rhs = zeros(ComplexF64, num_elements)
        rhsFill!(pulse_mesh1, fieldFunc, rhs)
        @test isapprox(rhs, rhs_solution, rtol=1e-15)
    end # rhsFill! tests

    @testset "rhsNormalDeriv! tests" begin
        amplitude = 1.0
        wavevector = [0.0, 0.0, 1/10+im*0]
        nodes_global = [0.0 0.0 1.5; 1.0 0.0 1.5; 0.0 1.0 1.5]
        area = 0.5
        elements = [1 2 3]
        num_elements = 1
        normals = [0.0 0.0 1.0]
        fieldFuncND(x,y,z,normal) = planeWaveNormalDerivative(amplitude, wavevector, [x,y,z], normal)
        # simple 1 integration point test 1 triangle
        pulse_mesh1 = PulseMesh(num_elements=num_elements, nodes=nodes_global, elements=elements, normals=normals,
                               areas=[area], test_quadrature_rule=gauss1rule,
                               test_quadrature_points=calculateQuadraturePoints(nodes_global, elements, gauss1rule[1:3,:]),
                               test_quadrature_weights=gauss1rule[4,:])
        rhs_solution = -area * planeWaveNormalDerivative(amplitude,
                                                         wavevector,
                                                         pulse_mesh1.test_quadrature_points[1][:],
                                                         normals[1,:])
        rhs = zeros(ComplexF64, num_elements)
        rhsNormalDerivFill!(pulse_mesh1, fieldFuncND, rhs)
        @test isapprox(rhs[1], rhs_solution, rtol=1e-15)

        amplitude = 1.0
        wavevector = [0.0, 0.0, 1/20+im*0]
        nodes_global = [0.0 0.0 1.0; 1.0 0.0 1.0; 0.0 1.0 1.0]
        area = 0.5
        elements = [1 2 3]
        num_elements = 1
        normals = [0.0 0.0 1.0]
        fieldFuncND(x,y,z,normal) = planeWaveNormalDerivative(amplitude, wavevector, [x,y,z], normal)
        # simple 1 integration point test 1 triangle
        pulse_mesh1 = PulseMesh(num_elements=num_elements, nodes=nodes_global, elements=elements, normals=normals,
                               areas=[area], test_quadrature_rule=gauss1rule,
                               test_quadrature_points=calculateQuadraturePoints(nodes_global, elements, gauss1rule[1:3,:]),
                               test_quadrature_weights=gauss1rule[4,:])
        rhs_solution = -area * planeWaveNormalDerivative(amplitude,
                                                         wavevector,
                                                         pulse_mesh1.test_quadrature_points[1][:],
                                                         normals[1,:])
        rhs = zeros(ComplexF64, num_elements)
        rhsNormalDerivFill!(pulse_mesh1, fieldFuncND, rhs)
        @test isapprox(rhs[1], rhs_solution, rtol=1e-15)
    end # rhsNormalDeriv! Tests

    @testset "matrixFill! tests" begin
        #first test is for non-singular elements with 1 point rule
        wavenumber = 1/10+im*0
        nodes_global = [0.0 0.0 1.5; 1.0 0.0 1.5; 0.0 1.0 1.5;
                     10.0 0.0 1.5; 11.0 0.0 1.5; 10.0 1.0 1.5]
        areas = [0.5, 0.5]
                normals = [0.0 0.0 1.0; 0.0 0.0 1.0]
        elements = [1 2 3; 4 5 6]
        num_elements = 2
        src_quadrature_rule = gauss7rule
        test_quadrature_rule = gauss1rule
        pulse_mesh = PulseMesh(num_elements, nodes_global, elements, areas, normals, src_quadrature_rule, test_quadrature_rule,
                  calculateQuadraturePoints(nodes_global, elements, src_quadrature_rule[1:3,:]), src_quadrature_rule[4,:],
                  calculateQuadraturePoints(nodes_global, elements, test_quadrature_rule[1:3,:]), test_quadrature_rule[4,:],
                  lu(ones(1,1)), [0])
        centroids = [1/3 1/3 1.5; 10.0+1/3 1/3 1.5]
        r_test = [10.0 + 1/3, 1/3, 1.5]
        distance_to_edge_tol = 1e-12
        near_singular_tol = 1.0
        matrix_solution = Array{Complex{Float64}, 2}(undef, num_elements, num_elements)
        matrix_solution[1,1] = areas[1] * scalarGreensSingularIntegral(wavenumber,
                                              centroids[1,:],
                                              nodes_global[1:3,:],
                                              gauss7rule[1:3,:],
                                              gauss7rule[4,:],
                                              distance_to_edge_tol)
        matrix_solution[2,2] = areas[2] * scalarGreensSingularIntegral(wavenumber,
                                              centroids[2,:],
                                              nodes_global[4:6,:],
                                              gauss7rule[1:3,:],
                                              gauss7rule[4,:],
                                              distance_to_edge_tol)
        greensFunc(x,y,z)=scalarGreens(norm([x,y,z]-centroids[2,:]), wavenumber)
        matrix_solution[2,1] = areas[2] * gaussQuadrature(areas[1], greensFunc, pulse_mesh.src_quadrature_points[1], pulse_mesh.src_quadrature_weights)
        greensFunc(x,y,z)=scalarGreens(norm([x,y,z]-centroids[1,:]), wavenumber)
        matrix_solution[1,2] = areas[1] * gaussQuadrature(areas[2], greensFunc, pulse_mesh.src_quadrature_points[2], pulse_mesh.src_quadrature_weights)
        testIntegrand(r_test, src_idx, is_singular) = scalarGreensIntegration(pulse_mesh,
                                                       src_idx,
                                                       wavenumber,
                                                       r_test,
                                                       distance_to_edge_tol,
                                                       near_singular_tol,
                                                       is_singular)
        z_matrix = zeros(ComplexF64, num_elements, num_elements)
        matrixFill!(pulse_mesh, testIntegrand, z_matrix)
        for src_idx in 1:num_elements
            for test_idx in 1:num_elements
                @test isapprox(z_matrix[test_idx, src_idx], matrix_solution[test_idx, src_idx], rtol=1e-15)
            end
        end

        #first test is for 3 triangles with 7 point rule
        wavenumber = 1/10+im*0
        nodes_global = [0.0 0.0 1.5; 1.0 0.0 1.5; 0.0 1.0 1.5;
                        10.0 0.0 1.5; 11.0 0.0 1.5; 10.0 1.0 1.5;
                        1.0 1.0 0.0]
        areas = [0.5, 0.5, 1.1726039399558574]
        normals = [ 0.0 0.0 1.0; 0.0 0.0 1.0; 0.639602 0.639602 0.426401]
        elements = [1 2 3; 4 5 6; 2 7 3]
        num_elements = 3
        src_quadrature_rule = gauss7rule
        test_quadrature_rule = gauss7rule
        pulse_mesh = PulseMesh(num_elements, nodes_global, elements, areas, normals, src_quadrature_rule, test_quadrature_rule,
                  calculateQuadraturePoints(nodes_global, elements, src_quadrature_rule[1:3,:]), src_quadrature_rule[4,:],
                  calculateQuadraturePoints(nodes_global, elements, test_quadrature_rule[1:3,:]), test_quadrature_rule[4,:],
                  lu(ones(1,1)), [0])
        centroids = [1/3 1/3 1.5; 10.0+1/3 1/3 1.5; 2/3 2/3 1.0]
        r_test = [10.0 + 1/3, 1/3, 1.5]
        distance_to_edge_tol = 1e-12
        near_singular_tol = 2.0
        matrix_solution = Array{Complex{Float64}, 2}(undef, num_elements, num_elements)
        integrand11(x,y,z) = scalarGreensSingularIntegral(wavenumber,
                                              [x,y,z], #r_test
                                              nodes_global[1:3,:],
                                              gauss7rule[1:3,:],
                                              gauss7rule[4,:],
                                              distance_to_edge_tol)
        scalar_greens_integrand(x,y,z) = scalarGreens(norm([x,y,z]-r_test), wavenumber)
        # The integrand below should evaluate to be purely numerical
        integrand12(x,y,z) = scalarGreensIntegration(pulse_mesh, 2, wavenumber,
                                         [x,y,z], #r_test
                                         distance_to_edge_tol,
                                         near_singular_tol,
                                         false) #is_singular

        integrand13(x,y,z) = scalarGreensNearSingularIntegral(wavenumber,
                                                  [x,y,z], #r_test
                                                  getTriangleNodes(3, elements, nodes_global),
                                                  areas[3],
                                                  pulse_mesh.src_quadrature_points[3],
                                                  pulse_mesh.src_quadrature_weights,
                                                  distance_to_edge_tol)
        matrix_solution[1,1] = gaussQuadrature(areas[1],
                                                 integrand11,
                                                 pulse_mesh.test_quadrature_points[1],
                                                 pulse_mesh.test_quadrature_weights)
        matrix_solution[1,2] = gaussQuadrature(areas[1],
                                                 integrand12,
                                                 pulse_mesh.test_quadrature_points[1],
                                                 pulse_mesh.test_quadrature_weights)
        matrix_solution[1,3] = gaussQuadrature(areas[1],
                                                 integrand13,
                                                 pulse_mesh.test_quadrature_points[1],
                                                 pulse_mesh.test_quadrature_weights)
        testIntegrand(r_test, src_idx, is_singular) = scalarGreensIntegration(pulse_mesh,
                                                       src_idx,
                                                       wavenumber,
                                                       r_test,
                                                       distance_to_edge_tol,
                                                       near_singular_tol,
                                                       is_singular)
        z_matrix = zeros(ComplexF64, num_elements, num_elements)
        matrixFill!(pulse_mesh, testIntegrand, z_matrix)
        @test isapprox(z_matrix[1,1], matrix_solution[1,1], rtol=1e-15)
        @test isapprox(z_matrix[1,2], matrix_solution[1,2], rtol=1e-15)
        @test isapprox(z_matrix[1,3], matrix_solution[1,3], rtol=1e-15)
        # next tests fill the matrix again, effectively doubling the entries
        matrixFill!(pulse_mesh, testIntegrand, z_matrix)
        @test isapprox(z_matrix[1,1], 2*matrix_solution[1,1], rtol=1e-15)
        @test isapprox(z_matrix[1,2], 2*matrix_solution[1,2], rtol=1e-15)
        @test isapprox(z_matrix[1,3], 2*matrix_solution[1,3], rtol=1e-15)
    end
    @testset "nodeMatrixFill! tests" begin
        wavenumber = 1.0+0.0im
        src_quadrature_rule = gauss7rule
        test_quadrature_rule = gauss7rule
        distance_to_edge_tol = 1e-12
        near_singular_tol = 1.0
        approximation_tol = 1e-3
        mesh_filename = "examples/test/rectangle_plate_8elements_symmetric.msh"
        pulse_mesh =  buildPulseMesh(mesh_filename, src_quadrature_rule, test_quadrature_rule)
        num_levels = 3
        octree = createOctree(num_levels, pulse_mesh)
        testIntegrand(r_test, src_idx, is_singular) = scalarGreensIntegration(pulse_mesh, src_idx,
                                                       wavenumber,
                                                       r_test,
                                                       distance_to_edge_tol,
                                                       near_singular_tol,
                                                       is_singular)
        z_matrix = zeros(ComplexF64, pulse_mesh.num_elements, pulse_mesh.num_elements)
        matrixFill!(pulse_mesh, testIntegrand, z_matrix)
        sol = z_matrix[1,2]
        test_node = octree.nodes[6]
        src_node = octree.nodes[7]
        sub_Z = zeros(ComplexF64, 1, 1)
        nodeMatrixFill!(pulse_mesh, test_node, src_node, testIntegrand, sub_Z)
        @test isapprox(sub_Z[1,1], sol, rtol=1e-14)

        sol = z_matrix[2,8]
        test_node = octree.nodes[7]
        src_node = octree.nodes[13]
        sub_Z = zeros(ComplexF64, 1, 1)
        nodeMatrixFill!(pulse_mesh, test_node, src_node, testIntegrand, sub_Z)
        @test isapprox(sub_Z[1,1], sol, rtol=1e-14)

        sol = z_matrix[[1,2],[3,4]]
        test_node = octree.nodes[2]
        src_node = octree.nodes[3]
        sub_Z = zeros(ComplexF64, 2, 2)
        nodeMatrixFill!(pulse_mesh, test_node, src_node, testIntegrand, sub_Z)
        @test isapprox(sub_Z, sol, rtol=1e-14)
    end # nodeMatrixFill! tests
end
