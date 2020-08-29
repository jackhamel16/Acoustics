using Test

include("../../src/greens_functions.jl")

@testset "greens_functions tests" begin
    @testset "scalar_greens tests" begin
        @test scalar_greens(1, 0) == 1/(4*pi)
        @test scalar_greens(1, 1) == exp(-im)/(4*pi)
        @test scalar_greens(-1.5, 2.1) == exp(-im*2.1*1.5)/(4*pi*1.5)
        @test isinf(scalar_greens(0, 100)) == true
    end
end
