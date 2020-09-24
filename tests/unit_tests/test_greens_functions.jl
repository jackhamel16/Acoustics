using Test
using LinearAlgebra

include("../../src/greens_functions.jl")

@testset "greens_functions tests" begin
    @testset "scalarGreens tests" begin
        @test scalarGreens(1.0, 0.0+0*im) == 1/(4*pi)
        @test scalarGreens(1.0, 1.0+0*im) == exp(-im)/(4*pi)
        @test scalarGreens(-1.5, 2.1+0*im) == exp(-im*2.1*1.5)/(4*pi*1.5)
        @test isinf(scalarGreens(0.0, 100.0+0*im)) == true
    end
    @testset "scalarGreens tests" begin
        @test scalarGreensNonSingular(1.0, 0.0+0*im) == 0.0
        @test scalarGreensNonSingular(1.0, 1.0+0*im) == (exp(-im)-1)/(4*pi)
        @test scalarGreensNonSingular(-1.5, 2.1+0*im) == (exp(-im*2.1*1.5)-1)/(4*pi*1.5)
        @test isnan(scalarGreensNonSingular(0.0, 100.0+0*im)) == true
    end
    @testset "singularScalarGreensIntegral tests" begin
        include("../../src/quadrature.jl")
        include("../../src/mesh.jl")
        # The following two tests compare singularScalarGreensIntegral to the result
        # of integrating 1/R using 7-point Gauss quadrature
        gauss7points_cartesian = Array{Float64, 2}(undef, 7, 3)
        nodes = [0.0 0.0 0.0; 2.0 0.0 0.0; 0.0 2.0 0.0]
        area = 2.0
        for point_idx in 1:7
            gauss7points_cartesian[point_idx,:] = barycentric2Cartesian(nodes, gauss7points[point_idx,:])
        end
        one_over_R(r_test, x, y, z) = 1 / norm(r_test - [x, y, z])
        distance_to_edge_tol = 1e-6
        r_test = [1000.0, 1.0, 0.0]
        integrand(x,y,z)=one_over_R(r_test,x,y,z)
        solution = gaussQuadrature(area, integrand, gauss7points_cartesian, gauss7weights)
        integral_results = singularScalarGreensIntegral(r_test, nodes, distance_to_edge_tol)
        @test isapprox(integral_results, solution, rtol=1e-6)

        r_test = [0.5, 0.5, -50.0]
        integrand(x,y,z)=one_over_R(r_test,x,y,z)
        solution = gaussQuadrature(area, integrand, gauss7points_cartesian, gauss7weights)
        integral_results = singularScalarGreensIntegral(r_test, nodes, distance_to_edge_tol)
        @test isapprox(integral_results, solution, rtol=1e-4)

        # This test is of when the projection of r_test is on one edge extension
        r_test = [1000.0, 0.0, 0.0]
        integrand(x,y,z)=one_over_R(r_test,x,y,z)
        solution = gaussQuadrature(area, integrand, gauss7points_cartesian, gauss7weights)
        integral_results = singularScalarGreensIntegral(r_test, nodes, distance_to_edge_tol)
        @test isapprox(integral_results, solution, rtol=1e-6)

        # This test is of when the projection of r_test is on one edge
        r_test = [1.0, 0.0, 1000.0]
        integrand(x,y,z)=one_over_R(r_test,x,y,z)
        solution = gaussQuadrature(area, integrand, gauss7points_cartesian, gauss7weights)
        integral_results = singularScalarGreensIntegral(r_test, nodes, distance_to_edge_tol)
        @test isapprox(integral_results, solution, rtol=1e-6)

        # This test is of when the projection of r_test is on a corner (two edges)
        r_test = [0.0, 2.0, 1000.0]
        integrand(x,y,z)=one_over_R(r_test,x,y,z)
        solution = gaussQuadrature(area, integrand, gauss7points_cartesian, gauss7weights)
        integral_results = singularScalarGreensIntegral(r_test, nodes, distance_to_edge_tol)
        @test isapprox(integral_results, solution, rtol=1e-6)

        # This test is of when the projection of r_test is really close to an edge
        r_test = [1e-12, 1.0, 1000.0]
        integrand(x,y,z)=one_over_R(r_test,x,y,z)
        solution = gaussQuadrature(area, integrand, gauss7points_cartesian, gauss7weights)
        integral_results = singularScalarGreensIntegral(r_test, nodes, distance_to_edge_tol)
        @test isapprox(integral_results, solution, rtol=1e-6)
    end
    @testset "computeSingularScalarGreensIntegralParameters tests" begin
        r_test = [2.0, 0.0, 1.0]
        nodes = [0.0 0.0 0.0; 2.0 0.0 0.0; 0.0 2.0 1.0]

        r_plus = circshift(nodes, (-1,0))
        r_minus = nodes
        n_hat = cross(r_plus[1,:]-r_minus[1,:],r_minus[3,:]-r_plus[3,:])/
                norm(cross(r_plus[1,:]-r_minus[1,:],r_minus[3,:]-r_plus[3,:]))
        d = dot(n_hat, r_test - r_plus[1,:])
        rho_test = r_test - d * n_hat

        rho_plus = Array{Float64, 2}(undef, 3, 3)
        rho_minus = Array{Float64, 2}(undef, 3, 3)
        I_hat = Array{Float64, 2}(undef, 3, 3)
        u_hat = Array{Float64, 2}(undef, 3, 3)
        l_plus = Array{Float64, 1}(undef, 3)
        l_minus = Array{Float64, 1}(undef, 3)
        P0 = Array{Float64, 1}(undef, 3)
        for i in 1:3 # i is triangle edge index
            rho_plus[i,:] = r_plus[i,:] - n_hat*dot(n_hat, r_plus[i,:])
            rho_minus[i,:] = r_minus[i,:] - n_hat*dot(n_hat, r_minus[i,:])
            I_hat[i,:] = (rho_plus[i,:]-rho_minus[i,:])/norm(rho_plus[i,:]-rho_minus[i,:])
            u_hat[i,:] = cross(I_hat[i,:], n_hat)
            l_plus[i] = dot(rho_plus[i,:]-rho_test, I_hat[i,:])
            l_minus[i] = dot(rho_minus[i,:]-rho_test, I_hat[i,:])
            P0[i] = abs(dot(rho_plus[i,:]-rho_test, u_hat[i,:]))
        end

        params = computeSingularScalarGreensIntegralParameters(r_test, nodes)
        @test params[1] == d
        @test params[1] == dot(n_hat, r_test - r_minus[1,:]) # another way to compute d
        for i in 1:3 # i is triangle edge index
            @test isapprox(params[2][i,:], (rho_plus[i,:] - rho_test - l_plus[i]*I_hat[i,:]) / P0[i]) # P0_hat
            @test isapprox(params[2][i,:], (rho_minus[i,:] - rho_test - l_minus[i]*I_hat[i,:]) / P0[i])  # another way to compute P0_hat for edge 2
            @test params[3][i,:] == u_hat[i,:]
            @test params[4][i] == P0[i]
            @test params[4][i] == abs(dot(rho_minus[i,:]-rho_test, u_hat[i,:])) #another way to compute P0
            @test params[5][i] == sqrt(P0[i]^2 + d^2) #R0
            @test params[6][i] == sqrt(norm(rho_plus[i,:]-rho_test)^2 + d^2) #R_plus
            @test params[7][i] == sqrt(norm(rho_minus[i,:]-rho_test)^2 + d^2) #R_minus
            @test params[8][i] == l_plus[i]
            @test params[9][i] == l_minus[i]
        end
    end
end
