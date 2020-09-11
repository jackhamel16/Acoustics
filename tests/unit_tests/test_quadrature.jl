using Test

include("../../src/quadrature.jl")

@testset "quadrature tests" begin
    @testset "gaussQuadrature tests" begin
        @testset "gaussQuadrature points and weights tests" begin
            gauss7points_reference = [3.33333333333333314829616256247391e-01 3.33333333333333314829616256247391e-01 3.33333333333333314829616256247391e-01;
                                      0.00000000000000000000000000000000e+00 0.00000000000000000000000000000000e+00 1.00000000000000000000000000000000e+00;
                                      0.00000000000000000000000000000000e+00 1.00000000000000000000000000000000e+00 0.00000000000000000000000000000000e+00;
                                      1.00000000000000000000000000000000e+00 0.00000000000000000000000000000000e+00 0.00000000000000000000000000000000e+00;
                                      5.00000000000000000000000000000000e-01 5.00000000000000000000000000000000e-01 0.00000000000000000000000000000000e+00;
                                      5.00000000000000000000000000000000e-01 0.00000000000000000000000000000000e+00 5.00000000000000000000000000000000e-01;
                                      0.00000000000000000000000000000000e+00 5.00000000000000000000000000000000e-01 5.00000000000000000000000000000000e-01]
            gauss7weights_reference = [4.50000000000000011102230246251565e-01
                                       5.00000000000000027755575615628914e-02
                                       5.00000000000000027755575615628914e-02
                                       5.00000000000000027755575615628914e-02
                                       1.33333333333333331482961625624739e-01
                                       1.33333333333333331482961625624739e-01
                                       1.33333333333333331482961625624739e-01]
            @test gauss7points == gauss7points_reference
            @test gauss7weights == gauss7weights_reference
        end
        @testset "gaussQuadrature function tests" begin
            returnZero(x,y,z) = 0
            points = [0.0 0.0 0.0]; weights = [1.0]
            @test gaussQuadrature(returnZero, points, weights) == 0
            returnXplusYplusZ(x,y,z) = x + y + z
            points = [-1.5 0.5 0.0; 0.0 1.0 2.0]; weights = [1.0, 9.1]
            solution = 26.299999999999997
            @test gaussQuadrature(returnXplusYplusZ, points, weights) == solution
            # This test will integrate over a triangle using the 7 point rule above
            f(x,y,z) = 2*x + y
            nodes = [0.0 0.0 0.0; 0.0 2.0 0.0; 2.0 0.0 0.0]
            gauss7points_cartesian = Array{Float64, 2}(undef, 7, 3)
            for point_idx in 1:7
                gauss7points_cartesian[point_idx,:] = barycentric2Cartesian(nodes, gauss7points[point_idx,:])
            end
            println(gauss7points_cartesian)
            solution = 20/3
            @test gaussQuadrature(f, gauss7points_cartesian, gauss7weights) == solution
        end
    end


end
