using Test

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
end
