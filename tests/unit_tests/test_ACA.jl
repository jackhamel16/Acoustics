using Test

include("../../src/quadrature.jl")
include("../../src/mesh.jl")
include("../../src/greens_functions.jl")
include("../../src/octree.jl")

include("../../src/ACA.jl")

@testset "ACA tests" begin
    @testset "computeRHSContribution tests" begin
        wavenumber = 1.0+0.0im
        src_quadrature_rule = gauss7rule
        test_quadrature_rule = gauss7rule
        distance_to_edge_tol = 1e-12
        near_singular_tol = 1.0
        mesh_filename = "examples/test/rectangle_plate_8elements_symmetric.msh"
        pulse_mesh =  buildPulseMesh(mesh_filename, src_quadrature_rule, test_quadrature_rule)
        testIntegrand(r_test, src_idx, is_singular) = scalarGreensIntegration(pulse_mesh, src_idx,
                                                       wavenumber,
                                                       r_test,
                                                       distance_to_edge_tol,
                                                       near_singular_tol,
                                                       is_singular)
        z_matrix = zeros(ComplexF64, pulse_mesh.num_elements, pulse_mesh.num_elements)
        matrixFill(pulse_mesh, testIntegrand, z_matrix)
        # centroids = Array{Array{Float64,1},1}(undef, pulse_mesh.num_elements)
        # for ele_idx = 1:pulse_mesh.num_elements
        #     centroids[ele_idx] = computeCentroid(pulse_mesh.nodes[pulse_mesh.elements[ele_idx,:],:])
        # end
        J_vec = randn(pulse_mesh.num_elements)
        V_vec = zeros(pulse_mesh.num_elements)
        num_levels = 1
        octree = createOctree(num_levels, pulse_mesh)
        node = octree.nodes[1]

        test = computeRHSContribution(pulse_mesh, wavenumber, distance_to_edge_tol, near_singular_tol, node, node, J_vec, V_vec)
        @test isapprox(test, z_matrix, rtol=1e-15)
    end #computeRHSContribution tests
end # ACA tests
