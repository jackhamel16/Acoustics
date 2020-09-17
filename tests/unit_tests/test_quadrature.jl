using Test

include("../../src/quadrature.jl")
include("../../src/mesh.jl")

@testset "quadrature tests" begin
    @testset "gaussQuadrature tests" begin
        @testset "gaussQuadrature points and weights tests" begin
            gauss7points_reference = [3.33333333333333314829616256247391e-01 3.33333333333333314829616256247391e-01 3.33333333333333314829616256247391e-01
                                      4.70142064105100010440452251714305e-01 4.70142064105100010440452251714305e-01 5.97158717897999791190954965713900e-01
                                      4.70142064105100010440452251714305e-01 5.97158717897999791190954965713900e-02 4.70142064105100010440452251714305e-01
                                      5.97158717897999791190954965713900e-02 4.70142064105100010440452251714305e-01 4.70142064105100010440452251714305e-01
                                      1.01286507323455995943639607048681e-01 1.01286507323455995943639607048681e-01 7.97426985353087980357145170273725e-01
                                      1.01286507323455995943639607048681e-01 7.97426985353087980357145170273725e-01 1.01286507323455995943639607048681e-01
                                      7.97426985353087980357145170273725e-01 1.01286507323455995943639607048681e-01 1.01286507323455995943639607048681e-01]
            gauss7weights_reference = [2.25000000000000255351295663786004e-01
                                       1.32394152788506136442236993389088e-01
                                       1.32394152788506136442236993389088e-01
                                       1.32394152788506136442236993389088e-01
                                       1.25939180544827139529573400977824e-01
                                       1.25939180544827139529573400977824e-01
                                       1.25939180544827139529573400977824e-01]
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
            f(x,y,z) = 2*x
            nodes = [0.0 0.0 0.0; 2.0 0.0 0.0; 0.0 2.0 0.0]
            gauss7points_cartesian = Array{Float64, 2}(undef, 7, 3)
            for point_idx in 1:7
                gauss7points_cartesian[point_idx,:] = barycentric2Cartesian(nodes, gauss7points[point_idx,:])
            end
            area = 2
            solution = 8/3/area
            @test gaussQuadrature(f, gauss7points_cartesian, gauss7weights) == solution
        end
    end


end
