using Test
using LinearAlgebra

include("../../src/quadrature.jl")
include("../../src/mesh.jl")

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
    @testset "scalarGreensSingularityIntegral tests" begin
        include("../../src/quadrature.jl")
        include("../../src/mesh.jl")
        # The following two tests compare scalarGreensSingularityIntegral to the result
        # of integrating 1/R using 7-point Gauss quadrature
        nodes = [0.0 0.0 0.0; 2.0 0.0 0.0; 0.0 2.0 0.0]
        one_over_R(r_test, x, y, z) = 1 / (4 * pi * norm(r_test - [x, y, z]))
        distance_to_edge_tol = 1e-6

        # Compare to an analytical solution
        solution = 0.07444950807488637/(4.0 * pi) # from mathematica Integrate[]
        nodes1 = [0.0 0.0 0.0; 1.0 0.0 0.0; 0.0 1.0 0.0]
        nodes2 = [1.0 0.0 0.0; 1.0 1.0 0.0; 0.0 1.0 0.0]
        r_test = [10.0, 10.0, 0.0]
        triangle1_results = scalarGreensSingularityIntegral(r_test, nodes1, distance_to_edge_tol)
        triangle2_results = scalarGreensSingularityIntegral(r_test, nodes2, distance_to_edge_tol)
        @test isapprox(triangle1_results+triangle2_results, solution, rtol=1e-13)
    
        quadrature_points1 = calculateQuadraturePoints(nodes, [1 2 3], gauss1rule[1:3,:])
        quadrature_points7 = calculateQuadraturePoints(nodes, [1 2 3], gauss7rule[1:3,:])
        quadrature_points13 = calculateQuadraturePoints(nodes, [1 2 3], gauss13rule[1:3,:])
        quadrature_points79 = calculateQuadraturePoints(nodes, [1 2 3], gauss79rule[1:3,:])
        r_test = [1000.0, 1.0, 0.0]
        integrand(x,y,z)=one_over_R(r_test,x,y,z)
        solution = integrateTriangle(nodes, integrand, quadrature_points7[1], gauss7rule[4,:])
        integral_results = scalarGreensSingularityIntegral(r_test, nodes, distance_to_edge_tol)
        @test isapprox(integral_results, solution, rtol=1e-6)

        r_test = [1000.0, 1.0, 0.0]
        integrand(x,y,z)=one_over_R(r_test,x,y,z)
        solution = integrateTriangle(nodes, integrand, quadrature_points13[1], gauss13rule[4,:])
        integral_results = scalarGreensSingularityIntegral(r_test, nodes, distance_to_edge_tol)
        @test isapprox(integral_results, solution, rtol=1e-6)

        r_test = [0.5, 0.5, -50.0]
        integrand(x,y,z)=one_over_R(r_test,x,y,z)
        solution = integrateTriangle(nodes, integrand, quadrature_points1[1], gauss1rule[4,:])
        integral_results = scalarGreensSingularityIntegral(r_test, nodes, distance_to_edge_tol)
        @test isapprox(integral_results, solution, rtol=1e-4)
        solution = integrateTriangle(nodes, integrand, quadrature_points7[1], gauss7rule[4,:])
        integral_results = scalarGreensSingularityIntegral(r_test, nodes, distance_to_edge_tol)
        @test isapprox(integral_results, solution, rtol=1e-11)
        solution = integrateTriangle(nodes, integrand, quadrature_points13[1], gauss13rule[4,:])
        integral_results = scalarGreensSingularityIntegral(r_test, nodes, distance_to_edge_tol)
        @test isapprox(integral_results, solution, rtol=1e-15)
        solution = integrateTriangle(nodes, integrand, quadrature_points79[1], gauss79rule[4,:])
        integral_results = scalarGreensSingularityIntegral(r_test, nodes, distance_to_edge_tol)
        @test isapprox(integral_results, solution, rtol=1e-15)

        # This test is of when the projection of r_test is on one edge extension
        r_test = [1000.0, 0.0, 0.0]
        integrand(x,y,z)=one_over_R(r_test,x,y,z)
        solution = integrateTriangle(nodes, integrand, quadrature_points7[1], gauss7rule[4,:])
        integral_results = scalarGreensSingularityIntegral(r_test, nodes, distance_to_edge_tol)
        @test isapprox(integral_results, solution, rtol=1e-6)

        # This test is of when the projection of r_test is on one edge
        r_test = [1.0, 0.0, 1000.0]
        integrand(x,y,z)=one_over_R(r_test,x,y,z)
        solution = integrateTriangle(nodes, integrand, quadrature_points7[1], gauss7rule[4,:])
        integral_results = scalarGreensSingularityIntegral(r_test, nodes, distance_to_edge_tol)
        @test isapprox(integral_results, solution, rtol=1e-6)

        # This test is of when the projection of r_test is on a corner (two edges)
        r_test = [0.0, 2.0, 1000.0]
        integrand(x,y,z)=one_over_R(r_test,x,y,z)
        solution = integrateTriangle(nodes, integrand, quadrature_points7[1], gauss7rule[4,:])
        integral_results = scalarGreensSingularityIntegral(r_test, nodes, distance_to_edge_tol)
        @test isapprox(integral_results, solution, rtol=1e-6)

        # This test is of when the projection of r_test is really close to an edge
        r_test = [1e-12, 1.0, 1000.0]
        integrand(x,y,z)=one_over_R(r_test,x,y,z)
        solution = integrateTriangle(nodes, integrand, quadrature_points7[1], gauss7rule[4,:])
        integral_results = scalarGreensSingularityIntegral(r_test, nodes, distance_to_edge_tol)
        @test isapprox(integral_results, solution, rtol=1e-6)

        # This test is of when the tolerance is too high
        r_test = [1.0, 100.0, 1000.0]
        high_tol = 10.0
        integrand(x,y,z)=one_over_R(r_test,x,y,z)
        solution = integrateTriangle(nodes, integrand, quadrature_points7[1], gauss7rule[4,:])
        integral_results = scalarGreensSingularityIntegral(r_test, nodes, high_tol)
        @test abs((integral_results - solution)/solution) > 0.5
    end

    @testset "scalarGreensNearSingularIntegral tests" begin
        # this test collapses the function to mimicing
        # scalarGreensSingularityIntegral via k=0
        wavenumber = 0.0+0.0*im
        r_test = [1000.0, 1.0, 0.0]
        nodes = [0.0 0.0 0.0; 2.0 0.0 0.0; 0.0 2.0 0.0]
        distance_to_edge_tol = 1e-12
        scalar_greens_integrand(x,y,z) = scalarGreens(norm(r_test-[x,y,z]), wavenumber)
        quadrature_points7 = calculateQuadraturePoints(nodes, [1 2 3], gauss7rule[1:3,:])
        solution = integrateTriangle(nodes, scalar_greens_integrand, quadrature_points7[1], gauss7rule[4,:])
        @test isapprox(scalarGreensNearSingularIntegral(wavenumber, r_test, nodes, quadrature_points7[1], gauss7rule[4,:], distance_to_edge_tol), solution, rtol=1e-7)

        # one edge length away from triangle
        wavenumber = 1/100 +0*im
        r_test = [1/3, 1/3, 1.0]
        nodes = [0.0 0.0 0.0; 1.0 0.0 0.0; 0.0 1.0 0.0]
        distance_to_edge_tol = 1e-12
        scalar_greens_integrand(x,y,z) = scalarGreens(norm([x,y,z]-r_test), wavenumber)
        quadrature_points7 = calculateQuadraturePoints(nodes, [1 2 3], gauss7rule[1:3,:])
        quadrature_points13 = calculateQuadraturePoints(nodes, [1 2 3], gauss13rule[1:3,:])
        quadrature_points79 = calculateQuadraturePoints(nodes, [1 2 3], gauss79rule[1:3,:])
        solution = integrateTriangle(nodes, scalar_greens_integrand, quadrature_points7[1], gauss7rule[4,:])
        @test isapprox(scalarGreensNearSingularIntegral(wavenumber, r_test, nodes, quadrature_points7[1], gauss7rule[4,:], distance_to_edge_tol), solution, rtol=1e-3)
        solution = integrateTriangle(nodes, scalar_greens_integrand, quadrature_points13[1], gauss13rule[4,:])
        @test isapprox(scalarGreensNearSingularIntegral(wavenumber, r_test, nodes, quadrature_points13[1], gauss13rule[4,:], distance_to_edge_tol), solution, rtol=1e-4)
        solution = integrateTriangle(nodes, scalar_greens_integrand, quadrature_points79[1], gauss79rule[4,:])
        @test isapprox(scalarGreensNearSingularIntegral(wavenumber, r_test, nodes, quadrature_points79[1], gauss79rule[4,:], distance_to_edge_tol), solution, rtol=1e-12)

        # ten edge lengths away from triangle
        wavenumber = 1/100 +0*im
        r_test = [1/3, 1/3, 10.0]
        nodes = [0.0 0.0 0.0; 1.0 0.0 0.0; 0.0 1.0 0.0]
        distance_to_edge_tol = 1e-12
        scalar_greens_integrand(x,y,z) = scalarGreens(norm([x,y,z]-r_test), wavenumber)
        solution = integrateTriangle(nodes, scalar_greens_integrand, quadrature_points7[1], gauss7rule[4,:])
        @test isapprox(scalarGreensNearSingularIntegral(wavenumber, r_test, nodes, quadrature_points7[1], gauss7rule[4,:], distance_to_edge_tol), solution, rtol=1e-9)
        solution = integrateTriangle(nodes, scalar_greens_integrand, quadrature_points13[1], gauss13rule[4,:])
        @test isapprox(scalarGreensNearSingularIntegral(wavenumber, r_test, nodes, quadrature_points13[1], gauss13rule[4,:], distance_to_edge_tol), solution, rtol=1e-12)
        solution = integrateTriangle(nodes, scalar_greens_integrand, quadrature_points79[1], gauss79rule[4,:])
        @test isapprox(scalarGreensNearSingularIntegral(wavenumber, r_test, nodes, quadrature_points79[1], gauss79rule[4,:], distance_to_edge_tol), solution, rtol=1e-15)

        # one edge length away from triangle in xy plane
        wavenumber = 1/10 +0*im
        r_test = [-1.0, 0.5, 0.0]
        nodes = [0.0 0.0 0.0; 1.0 0.0 0.0; 0.0 1.0 0.0]
        distance_to_edge_tol = 1e-12
        scalar_greens_integrand(x,y,z) = scalarGreens(norm([x,y,z]-r_test), wavenumber)
        solution = integrateTriangle(nodes, scalar_greens_integrand, quadrature_points7[1], gauss7rule[4,:])
        @test isapprox(scalarGreensNearSingularIntegral(wavenumber, r_test, nodes, quadrature_points7[1], gauss7rule[4,:], distance_to_edge_tol), solution, rtol=1e-4)
        solution = integrateTriangle(nodes, scalar_greens_integrand, quadrature_points13[1], gauss13rule[4,:])
        @test isapprox(scalarGreensNearSingularIntegral(wavenumber, r_test, nodes, quadrature_points13[1], gauss13rule[4,:], distance_to_edge_tol), solution, rtol=1e-5)
        solution = integrateTriangle(nodes, scalar_greens_integrand, quadrature_points79[1], gauss79rule[4,:])
        @test isapprox(scalarGreensNearSingularIntegral(wavenumber, r_test, nodes, quadrature_points79[1], gauss79rule[4,:], distance_to_edge_tol), solution, rtol=1e-14)

        # analytical test less than one edge lengths away from triangle
        mathematica_solution = 0.04149730535944524-im*0.00399171401828898 # skeptically trust to 12 digits
        wavenumber = 1/10+0*im
        r_test = [-0.5, 0.5, 0.5]
        nodes = [0.0 0.0 0.0; 1.0 0.0 0.0; 0.0 1.0 0.1]
        distance_to_edge_tol = 1e-12
        quadrature_points7 = calculateQuadraturePoints(nodes, [1 2 3], gauss7rule[1:3,:])
        quadrature_points13 = calculateQuadraturePoints(nodes, [1 2 3], gauss13rule[1:3,:])
        quadrature_points79 = calculateQuadraturePoints(nodes, [1 2 3], gauss79rule[1:3,:])
        @test isapprox(scalarGreensNearSingularIntegral(wavenumber, r_test, nodes,
              quadrature_points7[1], gauss7rule[4,:], distance_to_edge_tol), mathematica_solution, rtol=1e-7)
        @test isapprox(scalarGreensNearSingularIntegral(wavenumber, r_test, nodes,
              quadrature_points13[1], gauss13rule[4,:], distance_to_edge_tol), mathematica_solution, rtol=1e-8)
        @test isapprox(scalarGreensNearSingularIntegral(wavenumber, r_test, nodes,
              quadrature_points79[1], gauss79rule[4,:], distance_to_edge_tol), mathematica_solution, rtol=1e-14)


    end

    @testset "scalarGreensSingularIntegral tests" begin
        wavenumber = 1/10 +0*im
        r_test = [1/3, 1/3, 0.0]
        nodes = [0.0 0.0 0.0; 1.0 0.0 0.0; 0.0 1.0 0.0]
        distance_to_edge_tol = 1e-12
        quadrature_points7 = calculateQuadraturePoints(nodes, [1 2 3], gauss7rule[1:3,:])
        quadrature_points79 = calculateQuadraturePoints(nodes, [1 2 3], gauss79rule[1:3,:])
        solution_to_test_7point = scalarGreensSingularIntegral(wavenumber, r_test,
                                        nodes, gauss7rule[1:3,:], gauss7rule[4,:], distance_to_edge_tol)
        solution_to_test_79point = scalarGreensSingularIntegral(wavenumber, r_test,
                                        nodes, gauss79rule[1:3,:], gauss79rule[4,:], distance_to_edge_tol)

        sub_nodes = ([1/3 1/3 0.0; 1.0 0.0 0.0; 0.0 1.0 0.0],
                     [0.0 0.0 0.0; 1/3 1/3 0.0; 0.0 1.0 0.0],
                     [0.0 0.0 0.0; 1.0 0.0 0.0; 1/3 1/3 0.0])
        solution = 0.0
        for triangle_idx in 1:3
            quadrature_points7 = calculateQuadraturePoints(sub_nodes[triangle_idx], [1 2 3], gauss7rule[1:3,:])
            solution += scalarGreensNearSingularIntegral(wavenumber, r_test,
                                                         sub_nodes[triangle_idx],
                                                         quadrature_points7[1],
                                                         gauss7rule[4,:],
                                                         distance_to_edge_tol)
        end
        # Checks against the same algorithm, but done again above with
        # 79 point quadrature
        @test isapprox(solution_to_test_7point, solution, rtol=1e-14)
        #check against mathematic solution (trust to 12 digits with skepticism)
        # obtained with NIntegrate with AccuracyGoal -> 16, PrecisionGoal -> 12
        mathematica_solution = 0.19150130866-im*0.003978136822535155
        @test isapprox(solution_to_test_7point, mathematica_solution, rtol=1e-5)
        @test isapprox(solution_to_test_79point, mathematica_solution, rtol=1e-7)
    end

    @testset "scalarGreensIntegration tests" begin
        wavenumber = 1/10 + 0*im
        nodes = [0.0 0.0 0.0; 1.0 0.0 0.0; 0.0 -1.0 0.0]
        max_edge_length = sqrt(2)
        near_singular_tol = 2.0
        distance_to_edge_tol = 1e-12
        r_test_far = [0.5, 50.0, 0.0]
        r_test_near = [1/3, 1.0, 0.0]
        r_test_singular = [0.25, -0.25, 0.0]
        scalar_greens_integrand(x,y,z) = scalarGreens(norm([x,y,z]-r_test_far),
                                                      wavenumber)
        quadrature_points7 = calculateQuadraturePoints(nodes, [1 2 3], gauss7rule[1:3,:])
        solution_far = integrateTriangle(nodes, scalar_greens_integrand,
                                         quadrature_points7[1], gauss7rule[4,:])
        solution_near = scalarGreensNearSingularIntegral(wavenumber, r_test_near, nodes,
                                                         quadrature_points7[1], gauss7rule[4,:], distance_to_edge_tol)
        solution_singular = scalarGreensSingularIntegral(wavenumber, r_test_singular,
                                        nodes, gauss7rule[1:3,:], gauss7rule[4,:], distance_to_edge_tol)
        pulse_mesh = PulseMesh(nodes=nodes, elements=[1 2 3],
                               src_quadrature_rule=gauss7rule,
                               src_quadrature_points=quadrature_points7,
                               src_quadrature_weights=gauss7rule[4,:])
        @test isapprox(scalarGreensIntegration(pulse_mesh, 1, wavenumber,
                                               r_test_far, distance_to_edge_tol,
                                               near_singular_tol, false),
                       solution_far, rtol=1e-15)
        low_near_singular_tol = 4/(3*sqrt(2))-0.0001
        @test false == isapprox(scalarGreensIntegration(pulse_mesh, 1, wavenumber,
                                               r_test_near, distance_to_edge_tol,
                                               low_near_singular_tol, false),
                                solution_near, rtol=1e-15)
        @test isapprox(scalarGreensIntegration(pulse_mesh, 1, wavenumber,
                                               r_test_near, distance_to_edge_tol,
                                               near_singular_tol, false),
                       solution_near, rtol=1e-15)
        @test isapprox(scalarGreensIntegration(pulse_mesh, 1, wavenumber,
                                               r_test_singular, distance_to_edge_tol,
                                               near_singular_tol,
                                               true),
                       solution_singular, rtol=1e-15)
        @test false == isapprox(scalarGreensIntegration(pulse_mesh, 1, wavenumber,
                                               r_test_singular, distance_to_edge_tol,
                                               near_singular_tol, false),
                       solution_singular, rtol=1e-15)
    end

    @testset "computeScalarGreensSingularityIntegralParameters tests" begin
        r_test = [2.0, 0.0, 1.0]
        nodes = [0.0 0.0 0.0; 2.0 0.0 0.0; 0.0 2.0 1.0]

        r_plus = circshift(nodes, (-1,0))
        r_minus = nodes
        n_hat = cross(r_plus[1,:]-r_minus[1,:],r_minus[3,:]-r_plus[3,:])/
                norm(cross(r_plus[1,:]-r_minus[1,:],r_minus[3,:]-r_plus[3,:]))
        d = dot(n_hat, r_test - r_plus[1,:])
        rho_test = r_test - n_hat * dot(r_test, n_hat)

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

        params = computeScalarGreensSingularityIntegralParameters(r_test, nodes)
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
