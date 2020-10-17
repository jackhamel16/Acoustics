using Test
using LinearAlgebra

include("../../src/fill.jl")
include("../../src/quadrature.jl")
include("../../src/greens_functions.jl")

@testset "fill tests" begin
    @testset "rhsFill tests" begin
        planeWave(wavevector, r_test) = exp(-im*dot(wavevector, r_test))

        wavevector = [0.0, 0.0, 1/10]
        r_test = [1/3, 1/3, 1.5]
        nodes_global = [0.0 0.0 1.5; 1.0 0.0 1.5; 0.0 1.0 1.5]
        area = 0.5
        elements = [1 2 3]
        num_elements = 1
        fieldFunc(x,y,z) = planeWave(wavevector, [x,y,z])
        # simple 1 integration point test 1 triangle
        rhs_solution = area * planeWave(wavevector, r_test)
        rhs = rhsFill(num_elements, elements, nodes_global, fieldFunc, gauss1rule)
        @test isapprox(rhs[1], rhs_solution, rtol=1e-15)
        # 7 point integration test 1 triangle
        rhs_solution = integrateTriangle(nodes_global, fieldFunc, gauss7rule[:,1:3], gauss7rule[:,4])
        rhs = rhsFill(num_elements, elements, nodes_global, fieldFunc, gauss7rule)
        @test isapprox(rhs[1], rhs_solution, rtol=1e-15)

        # Next three tests involve two triangles
        wavevector = [0.0, 0.0, 1/10]
        nodes_global = [0.0 0.0 1.5; 1.0 0.0 1.5; 1.0 1.0 1.5; 0.0 1.0 1.5]
        areas = [0.5 0.5]
        elements = [1 2 4; 2 3 4]
        num_elements = 2
        fieldFunc(x,y,z) = planeWave(wavevector, [x,y,z])
        centroids = [1/3 1/3 1.5; 2/3 2/3 1.5]
        rhs_solution = [areas[1]*planeWave(wavevector, centroids[1,:]), areas[2]*planeWave(wavevector, centroids[2,:])]
        rhs = rhsFill(num_elements, elements, nodes_global, fieldFunc, gauss1rule)
        @test isapprox(rhs, rhs_solution, rtol=1e-15)

        wavevector = [0.0, 0.0, 1/10]
        nodes_global = [0.0 0.0 1.5; 1.0 0.0 1.5; 1.0 1.0 1.0; 0.0 1.0 1.5]
        areas = [0.5 0.6123724356957945]
        elements = [1 2 4; 2 3 4]
        num_elements = 2
        fieldFunc(x,y,z) = planeWave(wavevector, [x,y,z])
        centroids = [1/3 1/3 1.5; 2/3 2/3 1+1/3]
        rhs_solution = [areas[1]*planeWave(wavevector, centroids[1,:]), areas[2]*planeWave(wavevector, centroids[2,:])]
        rhs = rhsFill(num_elements, elements, nodes_global, fieldFunc, gauss1rule)
        @test isapprox(rhs, rhs_solution, rtol=1e-15)

        wavevector = [1/30, 1/20, 1/15]
        nodes_global = [0.0 0.0 1.5; 1.0 0.0 1.5; 1.0 1.0 1.0; 0.0 1.0 1.5]
        areas = [0.5 0.6123724356957945]
        elements = [1 2 4; 2 3 4]
        num_elements = 2
        fieldFunc(x,y,z) = planeWave(wavevector, [x,y,z])
        centroids = [1/3 1/3 1.5; 2/3 2/3 1+1/3]
        rhs_solution = [areas[1]*planeWave(wavevector, centroids[1,:]), areas[2]*planeWave(wavevector, centroids[2,:])]
        rhs = rhsFill(num_elements, elements, nodes_global, fieldFunc, gauss1rule)
        @test isapprox(rhs, rhs_solution, rtol=1e-15)
    end
    @testset "matrixFill tests" begin
        #first test is for non-singular elements with 1 point rule
        wavenumber = 1/10+im*0
        nodes_global = [0.0 0.0 1.5; 1.0 0.0 1.5; 0.0 1.0 1.5;
                     10.0 0.0 1.5; 11.0 0.0 1.5; 10.0 1.0 1.5]
        areas = [0.5, 0.5]
        elements = [1 2 3; 4 5 6]
        num_elements = 2
        centroids = [1/3 1/3 1.5; 10.0+1/3 1/3 1.5]
        r_test = [10.0 + 1/3, 1/3, 1.5]
        distance_to_edge_tol = 1e-12
        near_singular_tol = 1.0
        matrix_solution = Array{Complex{Float64}, 2}(undef, num_elements, num_elements)
        matrix_solution[1,1] = areas[1] * scalarGreensSingularIntegral(wavenumber,
                                              centroids[1,:],
                                              nodes_global[1:3,:],
                                              gauss7rule,
                                              distance_to_edge_tol)
        matrix_solution[2,2] = areas[2] * scalarGreensSingularIntegral(wavenumber,
                                              centroids[2,:],
                                              nodes_global[4:6,:],
                                              gauss7rule,
                                              distance_to_edge_tol)
        greensFunc(x,y,z)=scalarGreens(norm([x,y,z]-centroids[2,:]), wavenumber)
        matrix_solution[2,1] = areas[2] * integrateTriangle(nodes_global[1:3,:], greensFunc, gauss7rule[:,1:3], gauss7rule[:,4])
        greensFunc(x,y,z)=scalarGreens(norm([x,y,z]-centroids[1,:]), wavenumber)
        matrix_solution[1,2] = areas[1] * integrateTriangle(nodes_global[4:6,:], greensFunc, gauss7rule[:,1:3], gauss7rule[:,4])
        testIntegrand(r_test, nodes, is_singular) = scalarGreensIntegration(wavenumber,
                                                       r_test,
                                                       nodes,
                                                       gauss7rule,
                                                       distance_to_edge_tol,
                                                       near_singular_tol,
                                                       is_singular)
        matrix = matrixFill(num_elements, elements, nodes_global, testIntegrand, gauss1rule)
        for src_idx in 1:num_elements
            for test_idx in 1:num_elements
                @test isapprox(matrix[test_idx, src_idx], matrix_solution[test_idx, src_idx], rtol=1e-15)
            end
        end
    end
end
