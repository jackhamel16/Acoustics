using Test

include("../../src/excitation.jl")

@testset "excitation tests" begin
    @testset "planeWave tests" begin
        # reference solutions obtained in Mathematica with 16 digits of precision
        amplitude = 1.5
        wavevector = [0.0+0.0*im, 0.0+0.0*im, 0.0+0.0*im]
        position = [0.0, 0.0, 0.0]
        @test isapprox(planeWave(amplitude, wavevector, position), amplitude, rtol=1e-15)
        amplitude = 1.5
        wavevector = [0.0+0.0*im, 0.0+0.0*im, 62.0+0.0*im]
        position = [0.0, 0.0, 0.0]
        @test isapprox(planeWave(amplitude, wavevector, position), amplitude, rtol=1e-15)
        amplitude = -10.0
        wavevector = [0.0+0.0*im, 0.0+0.0*im, 1/15+0.0*im]
        position = [0.0, 0.0, 1.0]
        solution = -9.977786007011223 + im*0.666172949233930
        @test isapprox(planeWave(amplitude, wavevector, position), solution, rtol=1e-15)
        amplitude = 10.0
        wavevector = [0.0-0.01*im, 0.0+0.0*im, 1/15+0.0*im]
        position = [0.0, 10.0, -3.2]
        solution = 9.77330616178066 + im*2.117188387484732
        @test isapprox(planeWave(amplitude, wavevector, position), solution, rtol=1e-15)
    end
end
