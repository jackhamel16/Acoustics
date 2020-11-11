using Test
using GSL

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
        lmax = 1
        l = [1, 1, 1]
        m = [-1, 0, 1]
        Plm_idxs = [3, 2, 3]
        theta = pi/4
        phi = pi/8

        leg_polys, d_leg_polys = sf_legendre_deriv_array(GSL_SF_LEGENDRE_SPHARM, lmax, cos(theta))
        Plm = leg_polys[Plm_idxs]
        d_Plm = d_leg_polys[Plm_idxs]
        phase_factors = [1, 1, -1]
        exponentials = exp.(im.*m*phi)
        sph_harms_sol = phase_factors .* Plm .* exponentials
        dtheta_sph_harms_sol = -sin(theta) .* phase_factors .* d_Plm .* exponentials
        dphi_sph_harms_sol = 1im .* m .* phase_factors .* Plm .* exponentials

        sph_harms_test = Ylm_val(theta, phi, lmax)

        @test isapprox(sph_harms_test[1], sph_harms_sol, rtol=1e-15)
        @test isapprox(sph_harms_test[2], dtheta_sph_harms_sol, rtol=1e-15)
        @test isapprox(sph_harms_test[3], dphi_sph_harms_sol, rtol=1e-15)
        @test isapprox(sph_harms_test[4], l, rtol=1e-15)
        @test isapprox(sph_harms_test[5], m, rtol=1e-15)
    end
end
