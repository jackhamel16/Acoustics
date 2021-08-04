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
        amplitude = -1.1
        wavevector = [1.0-0.1*im, 0.0+0.0*im, 1/15+0.0*im]
        position = [5.0, 10.0, -3.2]
        solution = -0.049511307059018725 - im*0.6653440871389426
        @test isapprox(planeWave(amplitude, wavevector, position), solution, rtol=1e-15)
    end
    @testset "planeWaveNormalDerivative tests" begin
        # reference solutions obtained in Mathematica with 16 digits of precision
        amplitude = 1.5
        wavevector = [0.0+0.0*im, 0.0+0.0*im, 0.0+0.0*im]
        position = [0.0, 0.0, 0.0]
        normal = [0.0, 0.0, 0.0]
        @test isapprox(planeWaveNormalDerivative(amplitude, wavevector, position, normal), 0.0, rtol=1e-15)
        amplitude = 1.5
        wavevector = [0.0+0.0*im, 0.0+0.0*im, 62.0+0.0*im]
        position = [0.0, 0.0, 0.0]
        normal = [0.0, 0.0, 1.0]
        @test isapprox(planeWaveNormalDerivative(amplitude, wavevector, position, normal), -im*wavevector[3]*amplitude, rtol=1e-15)
        amplitude = 1.5
        wavevector = [1.0+0.0*im, 0.0+0.0*im, 0.0+0.0*im]
        position = [0.5, 0.0, 0.0]
        normal = [1.0, 0.0, 0.0]
        solution = -0.7191383079063045-im*1.3163738428355591
        @test isapprox(planeWaveNormalDerivative(amplitude, wavevector, position, normal), solution, rtol=1e-15)
        amplitude = 10.0
        wavevector = [1.0+0.0*im, 4.0+0.0*im, 1.0+0.0*im]
        position = [-1.0, 2.0, 1.5]
        normal = [1.0, 0.0, 0.0]
        solution = -7.984871126234903 + im*6.020119026848237
        @test isapprox(planeWaveNormalDerivative(amplitude, wavevector, position, normal), solution, rtol=1e-15)
        amplitude = 10.0
        wavevector = [1.0-1.0*im, 0.0+0.0*im, 0.0+0.0*im]
        position = [1.0, 0.0, 0.0]
        normal = [1.0, 0.0, 0.0]
        solution = -5.083259859995252 + im*1.1079376530669922
        @test isapprox(planeWaveNormalDerivative(amplitude, wavevector, position, normal), solution, rtol=1e-15)
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
    @testset "sphericalWaveNormalDerivative tests" begin
        # Solutions obtained from Mathematica
        near_zero = 1e-32 # sphericalBesselj is "singular" at x=0 though in the limit as x-> is non-singular
        r, theta, phi = 1.0, pi/2, 0.0
        position = [r*sin(theta)*cos(phi), r*sin(theta)*sin(phi), r*cos(theta)]
        wavenumber = 1.0
        amplitude = 2.0
        l, m = 0,0
        normal = [1.0, 0.0, 0.0]
        sol = -0.1699162315486493
        sph_wave_nd_test = sphericalWaveNormalDerivative(amplitude, wavenumber, position, l, m, normal)
        @test isapprox(sph_wave_nd_test, sol, rtol=1e-5)

        r, theta, phi = 1.0, pi/2, pi/2
        position = [r*sin(theta)*cos(phi), r*sin(theta)*sin(phi), r*cos(theta)]
        wavenumber = 1.0
        amplitude = 2.0
        l, m = 0,0
        normal = [0.0, 1.0, 0.0]
        sol = -0.1699162315486493
        sph_wave_nd_test = sphericalWaveNormalDerivative(amplitude, wavenumber, position, l, m, normal)
        @test isapprox(sph_wave_nd_test, sol, rtol=1e-5)

        position = [0.0, 1.0, 1.5]
        r = sqrt(sum(position.^2))
        theta, phi = acos(position[3] / r), atan(position[2], position[1])
        wavenumber = 1.0
        amplitude = 2.0
        l, m = 1,0
        normal = [0.0, 1.0, 1.0]
        sol = 0.03898203874226629
        sph_wave_nd_test = sphericalWaveNormalDerivative(amplitude, wavenumber, position, l, m, normal)
        @test isapprox(sph_wave_nd_test, sol, rtol=1e-14)

        position = [-0.25, 1/3, -2.5]
        r = sqrt(sum(position.^2))
        theta, phi = acos(position[3] / r), atan(position[2], position[1])
        wavenumber = 10.0
        amplitude = 2.0
        l, m = 3,1
        normal = [1.0, 1.0, 1.0]
        sol = 0.010978539926365383-im*0.09677011332756127
        sph_wave_nd_test = sphericalWaveNormalDerivative(amplitude, wavenumber, position, l, m, normal)
        @test isapprox(sph_wave_nd_test, sol, rtol=1e-14)

        position = [-0.25, 1/3, -2.5]
        r = sqrt(sum(position.^2))
        theta, phi = acos(position[3] / r), atan(position[2], position[1])
        wavenumber = 10.0
        amplitude = 2.0
        l, m = 3,3
        normal = [1.0, 1.0, 1.0]
        sol = -0.0013992912332537392+im*0.0008996991889269909
        sph_wave_nd_test = sphericalWaveNormalDerivative(amplitude, wavenumber, position, l, m, normal)
        @test isapprox(sph_wave_nd_test, sol, rtol=1e-14)
    end
    @testset "sphericalWaveKDerivative tests" begin
        wavelength = 10.0
        wavenumber = 2*pi/wavelength
        position = [1.0,0.0,0.0]
        l, m = 0, 0
        test = sphericalWaveKDerivative(wavenumber, position, l, m)
        solution = 0.456438961139459
        @test isapprox(solution, test, rtol=1e-14)

        wavelength = 4.2
        wavenumber = 2*pi/wavelength
        position = [1.0,0.25,-0.5]
        l, m = 2, -1
        test = sphericalWaveKDerivative(wavenumber, position, l, m)
        solution = -0.23792269069355165+im*0.05948067267338791
        @test isapprox(solution, test, rtol=1e-14)

        wavelength = 20.0
        wavenumber = 2*pi/wavelength
        deltak = 0.01 * wavenumber
        k_high = wavenumber + deltak/2
        k_low = wavenumber - deltak/2
        position = [1.0,0.25,-0.5]
        l, m = 2, 2
        field_high = sphericalWave(2*k_high, k_high, position, l, m)
        field_low = sphericalWave(2*k_low, k_low, position, l, m)
        approx_sol = (field_high - field_low) / deltak
        test = sphericalWaveKDerivative(wavenumber, position, l, m)
        @test isapprox(approx_sol, test, rtol=1e-5)
    end # sphericalWaveKDerivative tests
end
