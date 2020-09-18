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
    @testset "singularScalarGreens" begin
        d = -2.0
        P0_hat = [1.0 0 0; 0 1.0 0; 0 0 1.0]
        u_hat = [-1.0 0 0; 0 -1.0 0; 0 0 -1.0]
        P0 = [2.0, 1.0, 1.5]
        R0 = [1.2, 1.3, 1.4]
        R_plus = [3.0, 4.0, 1.0]
        R_minus = [1.0, 1.5, 2.1]
        l_plus = [0.5, 0.1, 0.5]
        l_minus = [0.2, 0.5, 0.2]
        solution = -2.09660929985
        @test isapprox(singularScalarGreens(d, P0_hat, u_hat, P0, R0, R_plus,
                                            R_minus, l_plus, l_minus), solution)
    end
    @testset "computeSingularScalarGreensParameters tests" begin
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
            l_plus[i] = dot(rho_test-r_plus[i,:], I_hat[i,:])
            l_minus[i] = dot(rho_test-r_minus[i,:], I_hat[i,:])
            P0[i] = abs(dot(rho_plus[i,:]-rho_test, u_hat[i,:]))
        end

        params = computeSingularScalarGreensParameters(r_test, nodes)
        @test params[1] == d
        @test params[1] == dot(n_hat, r_test - r_minus[1,:]) # another way to compute d
        for i in 1:3 # i is triangle edge index
            @test_skip params[2,i,:] == (dot(rho_plus[i,:]-rho_test, u_hat[i,:]) - l_plus[i]*I_hat[i,:]) / P0 # P0_hat
            @test_skip params[2,i,:] == (dot(rho_minus[i,:]-rho_test, u_hat[i,:]) - l_minus[i]*I_hat[i,:]) / P0  # another way to compute P0_hat for edge 2
            @test_skip params[3,i,:] == u_hat[i,:]
            @test_skip params[4,i] == P0[i]
            @test_skip params[4,i] == abs(dot(rho_minus[i,:]-rho_test, u_hat[i,:])) #another way to compute P0
            @test_skip params[5,i] == sqrt(P0[i]^2 + d^2) #R0
            @test_skip params[6,i] == sqrt(P_plus[i]^2 + d^2) #R_plus
            @test_skip params[7,i] == sqrt(P_minus[i]^2 + d^2) #R_minus
            @test_skip params[8,i] == l_plus[i]
            @test_skip params[9,i] == l_minus[i]
        end
    end
end
