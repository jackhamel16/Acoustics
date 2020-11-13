using Test
using GSL

include("../../src/math.jl")

@testset "math tests" begin
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
    @testset "sphericalBesselj tests" begin
        x = 0.5
        j0 = sin(x)/x
        j1 = sin(x)/x^2-cos(x)/x
        j3 = (15/x^3-6/x)*sin(x)/x - (15/x^2-1)*cos(x)/x

        @test isapprox(sphericalBesselj(0.0, x), j0, rtol=1e-15)
        @test isapprox(sphericalBesselj(1.0, x), j1, rtol=1e-14)
        @test isapprox(sphericalBesselj(3.0, x), j3, rtol=1e-10)
    end
end
