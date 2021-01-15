using Test
using GSL

include("../../src/math.jl")

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
    @testset "sphericalWave tests" begin
        near_zero = 1e-32 # sphericalBesselj is "singular" at x=0 though in the limit as x-> is non-singular
        r, theta, phi = 1.1, pi/3, pi/4
        position = [r*sin(theta)*cos(phi), r*sin(theta)*sin(phi), r*cos(theta)]
        wavenumber = 2*pi/10
        amplitude = 2.0
        l, m = 0, 0

        sol = amplitude * sphericalHarmonics(theta, phi, l)[1][1] * sphericalBesselj(l, wavenumber * r)
        sph_wave_test = sphericalWave(amplitude, wavenumber, position, l, m)
        @test isapprox(sph_wave_test, sol, rtol=1e-15)

        r, theta, phi = near_zero, pi/2, 0.0
        position = [r*sin(theta)*cos(phi), r*sin(theta)*sin(phi), r*cos(theta)]
        wavenumber = 1.0
        amplitude = 1.0
        l, m = 1, 0

        sol = 0.0+0.0*im
        sph_wave_test = sphericalWave(amplitude, wavenumber, position, l, m)
        @test isapprox(sph_wave_test, sol, atol=1e-14)

        r, theta, phi = 1.1, pi/3, pi/4
        position = [r*sin(theta)*cos(phi), r*sin(theta)*sin(phi), r*cos(theta)]
        wavenumber = 2*pi/10
        amplitude = 2.0
        l, m = 3, -1

        sol = amplitude * sphericalHarmonics(theta, phi, l)[1][12] * sphericalBesselj(l, wavenumber * r)
        sph_wave_test = sphericalWave(amplitude, wavenumber, position, l, m)
        @test isapprox(sph_wave_test, sol, rtol=1e-15)

        r, theta, phi = 20.0, pi/3, -pi/4
        position = [r*sin(theta)*cos(phi), r*sin(theta)*sin(phi), r*cos(theta)]
        wavenumber = 2*pi/100
        amplitude = -0.62
        l, m = 2, 0

        sol = amplitude * sphericalHarmonics(theta, phi, l)[1][7] * sphericalBesselj(l, wavenumber * r)
        sph_wave_test = sphericalWave(amplitude, wavenumber, position, l, m)
        @test isapprox(sph_wave_test, sol, rtol=1e-15)
    end
end
