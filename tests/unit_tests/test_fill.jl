using Test
using LinearAlgebra

include("../../src/fill.jl")
include("../../src/quadrature.jl")

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
end
