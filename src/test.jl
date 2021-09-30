# using LinearAlgebra
# using IterativeSolvers
# using LinearMaps
using Plots

include("includes.jl")
# radius = 1
# # height = 2
# num_elements = 1180
# #
# area = 2*(2 * pi*radius^2 + 2*pi*radius*height) #doubled for slotted cyl
# area = 4* pi *radius^2 # sphere
# # r = 1
# # area = pi * r^2 #plate
# println("Area: ", area)
#
# ele_area = area / num_elements
# avg_edge_length = sqrt(4 * ele_area / sqrt(3))
# println(avg_edge_length)
# lambda = 10*avg_edge_length
# println("min lambda = ",lambda)
# max_l = ceil(height*2*pi/lambda)
# println("max degree, l = ", max_l)
# num_harmonics = max_l^2 + 2*max_l + 1
# println("num harmonics = ", num_harmonics)

excitation_amplitude = 1.0
lambda=2.5
wavenumber = 2*pi/lambda + 0*im
wavevector = [wavenumber, 0.0, wavenumber] ./ sqrt(2)
src_quadrature_rule = gauss7rule
test_quadrature_rule = gauss1rule
distance_to_edge_tol = 1e-12
near_singular_tol = 1.0

# mesh_filename = "examples/test/circular_plate_1m.msh"
# mesh_filename = "examples/test/spheres/sphere_1m_1266.msh"
#
# # sphere with plane wave
# planeWaveExcitation(x_test, y_test, z_test) = planeWave(excitation_amplitude, wavevector, [x_test,y_test,z_test])
# planeWaveExcitationNormalDeriv(x_test, y_test, z_test, normal) = planeWaveNormalDerivative(excitation_amplitude, wavevector, [x_test,y_test,z_test], normal)
#
# pulse_mesh = buildPulseMesh(mesh_filename, src_quadrature_rule, test_quadrature_rule)
# sourcesIE = solveSoftIE(pulse_mesh,
#                 planeWaveExcitation,
#                 wavenumber,
#                 distance_to_edge_tol,
#                 near_singular_tol)
# pulse_mesh = buildPulseMesh(mesh_filename, src_quadrature_rule, test_quadrature_rule)
# sourcesIEnd = solveSoftIENormalDeriv(pulse_mesh,
#                 planeWaveExcitationNormalDeriv,
#                 wavenumber)
# soft_IE_only = 1.0
# soft_IE_nd_only = 0.0
# pulse_mesh = buildPulseMesh(mesh_filename, src_quadrature_rule, test_quadrature_rule)
# sourcesCFIE_soft_IE_only = solveSoftCFIE(pulse_mesh,
#                 planeWaveExcitation,
#                 planeWaveExcitationNormalDeriv,
#                 wavenumber,
#                 distance_to_edge_tol,
#                 near_singular_tol,
#                 soft_IE_only)
# pulse_mesh = buildPulseMesh(mesh_filename, src_quadrature_rule, test_quadrature_rule)
# sourcesCFIE_soft_IE_nd_only = solveSoftCFIE(pulse_mesh,
#                 planeWaveExcitation,
#                 planeWaveExcitationNormalDeriv,
#                 wavenumber,
#                 distance_to_edge_tol,
#                 near_singular_tol,
#                 soft_IE_nd_only)
# isapprox(sourcesCFIE_soft_IE_only, sourcesIE, rtol=1e-12)
# # @test false==isapprox(sourcesCFIE_soft_IE_only, sourcesIEnd, rtol=1e-12)
# isapprox(sourcesCFIE_soft_IE_only, sourcesIEnd, rtol=1e-2)
# isapprox(sourcesCFIE_soft_IE_nd_only, sourcesIEnd, rtol=1e-12)
# # @test false==isapprox(sourcesCFIE_soft_IE_nd_only, sourcesIE, rtol=1e-12)
# isapprox(sourcesCFIE_soft_IE_nd_only, sourcesIE, rtol=1e-2)

# sphere with spherical wave
######################
mesh_filename = "examples/test/spheres/sphere_1m_1266.msh"
l, m = 0, 0
sphericalWaveExcitation(x_test, y_test, z_test) = sphericalWave(excitation_amplitude, wavenumber, [x_test,y_test,z_test], l, m)
sphericalWaveExcitationNormalDeriv(x_test, y_test, z_test, normal) = sphericalWaveNormalDerivative(excitation_amplitude, wavenumber, [x_test,y_test,z_test], l, m, normal)

pulse_meshIE = buildPulseMesh(mesh_filename, src_quadrature_rule, test_quadrature_rule)
sourcesIE = solveSoftIE(pulse_meshIE,
                sphericalWaveExcitation,
                wavenumber,
                distance_to_edge_tol,
                near_singular_tol)
pulse_meshIEnd = buildPulseMesh(mesh_filename, src_quadrature_rule, test_quadrature_rule)
sourcesIEnd = solveSoftIENormalDeriv(pulse_meshIEnd,
                sphericalWaveExcitationNormalDeriv,
                wavenumber)
soft_IE_only = 1.0
soft_IE_nd_only = 0.0
pulse_meshCFIE_IE = buildPulseMesh(mesh_filename, src_quadrature_rule, test_quadrature_rule)
sourcesCFIE_soft_IE_only = solveSoftCFIE(pulse_meshCFIE_IE,
                sphericalWaveExcitation,
                sphericalWaveExcitationNormalDeriv,
                wavenumber,
                distance_to_edge_tol,
                near_singular_tol,
                soft_IE_only)
pulse_meshCFIE_IEnd = buildPulseMesh(mesh_filename, src_quadrature_rule, test_quadrature_rule)
sourcesCFIE_soft_IE_nd_only = solveSoftCFIE(pulse_meshCFIE_IEnd,
                sphericalWaveExcitation,
                sphericalWaveExcitationNormalDeriv,
                wavenumber,
                distance_to_edge_tol,
                near_singular_tol,
                soft_IE_nd_only)
# isapprox(sourcesIE, sourcesIEnd, rtol=0.11e-1)
# isapprox(sourcesCFIE_soft_IE_only, sourcesIE, rtol=1e-12)
# # @test false==isapprox(sourcesCFIE_soft_IE_only, sourcesIEnd, rtol=1e-12)
# isapprox(sourcesCFIE_soft_IE_only, sourcesIEnd, rtol=1e-2)
# isapprox(sourcesCFIE_soft_IE_nd_only, sourcesIEnd, rtol=1e-12)
# # @test false==isapprox(sourcesCFIE_soft_IE_nd_only, sourcesIE, rtol=1e-12)
# isapprox(sourcesCFIE_soft_IE_nd_only, sourcesIE, rtol=0.11e-1)

println("\nQ1: Is the IE equal to IE normal deriv with l=m=0?")
l2norm_error = norm(sourcesIE .- sourcesIEnd) / norm(sourcesIE)
println("Q1 Error = ", l2norm_error)

println("\nQ2: what happens if I compute ZJ=V where Z and V are using the soft IE and J comes from soft IE normal deriv?")
test_V = pulse_meshIE.Z_factors.L * pulse_meshIE.Z_factors.U * sourcesIEnd
l2norm_error = norm(pulse_meshIE.RHS .- test_V) / norm(pulse_meshIE.RHS)
println("Q2 Error = ", l2norm_error)

println("\nQ3: Is the IE equal to IE normal deriv with l,m=5,-3?")
l, m = 5, -3
sphericalWaveExcitation(x_test, y_test, z_test) = sphericalWave(excitation_amplitude, wavenumber, [x_test,y_test,z_test], l, m)
sphericalWaveExcitationNormalDeriv(x_test, y_test, z_test, normal) = sphericalWaveNormalDerivative(excitation_amplitude, wavenumber, [x_test,y_test,z_test], l, m, normal)
pulse_meshIE = buildPulseMesh(mesh_filename, src_quadrature_rule, test_quadrature_rule)
sourcesIE = solveSoftIE(pulse_meshIE,
                sphericalWaveExcitation,
                wavenumber,
                distance_to_edge_tol,
                near_singular_tol)
pulse_meshIEnd = buildPulseMesh(mesh_filename, src_quadrature_rule, test_quadrature_rule)
sourcesIEnd = solveSoftIENormalDeriv(pulse_meshIEnd,
                sphericalWaveExcitationNormalDeriv,
                wavenumber)
l2norm_error = norm(sourcesIE .- sourcesIEnd) / norm(sourcesIE)
println("Q3 Error = ", l2norm_error)

println("\nQ4: Is the IE equal to IE normal deriv with l,m=8, 5?")
l, m = 8,5
sphericalWaveExcitation(x_test, y_test, z_test) = sphericalWave(excitation_amplitude, wavenumber, [x_test,y_test,z_test], l, m)
sphericalWaveExcitationNormalDeriv(x_test, y_test, z_test, normal) = sphericalWaveNormalDerivative(excitation_amplitude, wavenumber, [x_test,y_test,z_test], l, m, normal)
pulse_meshIE = buildPulseMesh(mesh_filename, src_quadrature_rule, test_quadrature_rule)
sourcesIE = solveSoftIE(pulse_meshIE,
                sphericalWaveExcitation,
                wavenumber,
                distance_to_edge_tol,
                near_singular_tol)
pulse_meshIEnd = buildPulseMesh(mesh_filename, src_quadrature_rule, test_quadrature_rule)
sourcesIEnd = solveSoftIENormalDeriv(pulse_meshIEnd,
                sphericalWaveExcitationNormalDeriv,
                wavenumber)
l2norm_error = norm(sourcesIE .- sourcesIEnd) / norm(sourcesIE)
println("Q4 Error = ", l2norm_error)
println("Based on Q1, 3 and 4, it looks like the more spacial variation in the excitation there is, the worse the error is.")

println("\nQ5: How does the soft IE compare to the soft IE normal deriv when the incident field is a planewave?")
planeWaveExcitation(x_test, y_test, z_test) = planeWave(excitation_amplitude, wavevector, [x_test,y_test,z_test])
planeWaveExcitationNormalDeriv(x_test, y_test, z_test, normal) = planeWaveNormalDerivative(excitation_amplitude, wavevector, [x_test,y_test,z_test], normal)
pulse_meshIE = buildPulseMesh(mesh_filename, src_quadrature_rule, test_quadrature_rule)
sourcesIE = solveSoftIE(pulse_meshIE,
                planeWaveExcitation,
                wavenumber,
                distance_to_edge_tol,
                near_singular_tol)
pulse_meshIEnd = buildPulseMesh(mesh_filename, src_quadrature_rule, test_quadrature_rule)
sourcesIEnd = solveSoftIENormalDeriv(pulse_meshIEnd,
                planeWaveExcitationNormalDeriv,
                wavenumber)
l2norm_error = norm(sourcesIE .- sourcesIEnd) / norm(sourcesIE)
println("Q5 Error = ", l2norm_error)

println("\nQ6: How does the soft IE compare to the soft IE normal deriv when the mesh is a cube w/ spherical wave?")
mesh_filename = "examples/simple/cube/cube_1m.msh"
l, m = 0, 0
sphericalWaveExcitation(x_test, y_test, z_test) = sphericalWave(excitation_amplitude, wavenumber, [x_test,y_test,z_test], l, m)
sphericalWaveExcitationNormalDeriv(x_test, y_test, z_test, normal) = sphericalWaveNormalDerivative(excitation_amplitude, wavenumber, [x_test,y_test,z_test], l, m, normal)
pulse_meshIE = buildPulseMesh(mesh_filename, src_quadrature_rule, test_quadrature_rule)
sourcesIE = solveSoftIE(pulse_meshIE,
                sphericalWaveExcitation,
                wavenumber,
                distance_to_edge_tol,
                near_singular_tol)
pulse_meshIEnd = buildPulseMesh(mesh_filename, src_quadrature_rule, test_quadrature_rule)
sourcesIEnd = solveSoftIENormalDeriv(pulse_meshIEnd,
                sphericalWaveExcitationNormalDeriv,
                wavenumber)
l2norm_error = norm(sourcesIE .- sourcesIEnd) / norm(sourcesIE)
println("Q6 Error = ", l2norm_error)

println("\nQ7: How does the soft IE compare to the soft IE normal deriv when the mesh is a cylinder w/ spherical end caps w/ spherical wave?")
mesh_filename = "examples/simple/cylinder_with_spherical_caps_1841.msh"
l, m = 0, 0
sphericalWaveExcitation(x_test, y_test, z_test) = sphericalWave(excitation_amplitude, wavenumber, [x_test,y_test,z_test], l, m)
sphericalWaveExcitationNormalDeriv(x_test, y_test, z_test, normal) = sphericalWaveNormalDerivative(excitation_amplitude, wavenumber, [x_test,y_test,z_test], l, m, normal)
pulse_meshIE = buildPulseMesh(mesh_filename, src_quadrature_rule, test_quadrature_rule)
sourcesIE = solveSoftIE(pulse_meshIE,
                sphericalWaveExcitation,
                wavenumber,
                distance_to_edge_tol,
                near_singular_tol)
pulse_meshIEnd = buildPulseMesh(mesh_filename, src_quadrature_rule, test_quadrature_rule)
sourcesIEnd = solveSoftIENormalDeriv(pulse_meshIEnd,
                sphericalWaveExcitationNormalDeriv,
                wavenumber)
exportSourcesGmsh(mesh_filename, "sourcesIE_cyl_sph_caps_real", real.(sourcesIE))
exportSourcesGmsh(mesh_filename, "sourcesIE_cyl_sph_caps_mag", abs.(sourcesIE))
exportSourcesGmsh(mesh_filename, "sourcesIEnd_cyl_sph_caps_real", real.(sourcesIEnd))
exportSourcesGmsh(mesh_filename, "sourcesIEnd_cyl_sph_caps_mag", abs.(sourcesIEnd))
exportSourcesGmsh(mesh_filename, "scale_factor_cyl_sph_caps_real", real.(sourcesIEnd ./ sourcesIE))
exportSourcesGmsh(mesh_filename, "scale_factor_cyl_sph_caps_mag", abs.(sourcesIEnd ./ sourcesIE))
exportSourcesGmsh(mesh_filename, "rel_error_cyl_sph_caps_real", real.((sourcesIE - sourcesIEnd) ./ sourcesIE))
exportSourcesGmsh(mesh_filename, "rel_error_cyl_sph_caps_mag", abs.((sourcesIE - sourcesIEnd) ./ sourcesIE))
l2norm_error = norm(sourcesIE .- sourcesIEnd) / norm(sourcesIE)
println("Q7 Error = ", l2norm_error)

println("\nQ8: How does the soft IE compare to the soft IE normal deriv when the mesh is a cylinder w/ spherical end caps w/ plane wave?")
mesh_filename = "examples/simple/cylinder_with_spherical_caps_1841.msh"
lambda=2.0
wavenumber = 2*pi/lambda + 0*im
wavevector = [0.0, 0.0, wavenumber]
planeWaveExcitation(x_test, y_test, z_test) = planeWave(excitation_amplitude, wavevector, [x_test,y_test,z_test])
planeWaveExcitationNormalDeriv(x_test, y_test, z_test, normal) = planeWaveNormalDerivative(excitation_amplitude, wavevector, [x_test,y_test,z_test], normal)
pulse_meshIE = buildPulseMesh(mesh_filename, src_quadrature_rule, test_quadrature_rule)
sourcesIE = solveSoftIE(pulse_meshIE,
                planeWaveExcitation,
                wavenumber,
                distance_to_edge_tol,
                near_singular_tol)
pulse_meshIEnd = buildPulseMesh(mesh_filename, src_quadrature_rule, test_quadrature_rule)
sourcesIEnd = solveSoftIENormalDeriv(pulse_meshIEnd,
                planeWaveExcitationNormalDeriv,
                wavenumber)
exportSourcesGmsh(mesh_filename, "sourcesIE_cyl_sph_caps_real", real.(sourcesIE))
exportSourcesGmsh(mesh_filename, "sourcesIE_cyl_sph_caps_mag", abs.(sourcesIE))
exportSourcesGmsh(mesh_filename, "sourcesIEnd_cyl_sph_caps_real", real.(sourcesIEnd))
exportSourcesGmsh(mesh_filename, "sourcesIEnd_cyl_sph_caps_mag", abs.(sourcesIEnd))
exportSourcesGmsh(mesh_filename, "scale_factor_cyl_sph_caps_real", real.(sourcesIEnd ./ sourcesIE))
exportSourcesGmsh(mesh_filename, "scale_factor_cyl_sph_caps_mag", abs.(sourcesIEnd ./ sourcesIE))
exportSourcesGmsh(mesh_filename, "rel_error_cyl_sph_caps_mag", abs.((sourcesIE - sourcesIEnd) ./ sourcesIE))
l2norm_error = norm(sourcesIE .- sourcesIEnd) / norm(sourcesIE)
println("Q8 Error = ", l2norm_error)


println("\nQ9: How does the soft IE compare to the soft IE normal deriv when the mesh is a very short cylinder w/ spherical end caps w/ spherical wave?")
mesh_filename = "examples/simple/cylinder_short_with_spherical_caps_1275.msh"
l, m = 0, 0
lambda=2.0
wavenumber = 2*pi/lambda + 0*im
sphericalWaveExcitation(x_test, y_test, z_test) = sphericalWave(excitation_amplitude, wavenumber, [x_test,y_test,z_test], l, m)
sphericalWaveExcitationNormalDeriv(x_test, y_test, z_test, normal) = sphericalWaveNormalDerivative(excitation_amplitude, wavenumber, [x_test,y_test,z_test], l, m, normal)
pulse_meshIE = buildPulseMesh(mesh_filename, src_quadrature_rule, test_quadrature_rule)
sourcesIE = solveSoftIE(pulse_meshIE,
                sphericalWaveExcitation,
                wavenumber,
                distance_to_edge_tol,
                near_singular_tol)
pulse_meshIEnd = buildPulseMesh(mesh_filename, src_quadrature_rule, test_quadrature_rule)
sourcesIEnd = solveSoftIENormalDeriv(pulse_meshIEnd,
                sphericalWaveExcitationNormalDeriv,
                wavenumber)
exportSourcesGmsh(mesh_filename, "sourcesIE_shcyl_sph_caps_real", real.(sourcesIE))
exportSourcesGmsh(mesh_filename, "sourcesIE_shcyl_sph_caps_mag", abs.(sourcesIE))
exportSourcesGmsh(mesh_filename, "sourcesIEnd_shcyl_sph_caps_real", real.(sourcesIEnd))
exportSourcesGmsh(mesh_filename, "sourcesIEnd_shcyl_sph_caps_mag", abs.(sourcesIEnd))
exportSourcesGmsh(mesh_filename, "scale_factor_shcyl_sph_caps_real", real.(sourcesIEnd ./ sourcesIE))
exportSourcesGmsh(mesh_filename, "scale_factor_shcyl_sph_caps_mag", abs.(sourcesIEnd ./ sourcesIE))
exportSourcesGmsh(mesh_filename, "rel_error_shcyl_sph_caps_real", real.((sourcesIE - sourcesIEnd) ./ sourcesIE))
exportSourcesGmsh(mesh_filename, "rel_error_shcyl_sph_caps_mag", abs.((sourcesIE - sourcesIEnd) ./ sourcesIE))
l2norm_error = norm(sourcesIE .- sourcesIEnd) / norm(sourcesIE)
println("Q9 Error = ", l2norm_error)

println("\nQ10: How does the soft IE compare to the soft IE normal deriv when the mesh is a near zero height cylinder w/ spherical end caps w/ spherical wave?")
mesh_filename = "examples/simple/cylinder_nearzero_with_spherical_caps_1887.msh"
l, m = 0, 0
lambda=2.0
wavenumber = 2*pi/lambda + 0*im
sphericalWaveExcitation(x_test, y_test, z_test) = sphericalWave(excitation_amplitude, wavenumber, [x_test,y_test,z_test], l, m)
sphericalWaveExcitationNormalDeriv(x_test, y_test, z_test, normal) = sphericalWaveNormalDerivative(excitation_amplitude, wavenumber, [x_test,y_test,z_test], l, m, normal)
pulse_meshIE = buildPulseMesh(mesh_filename, src_quadrature_rule, test_quadrature_rule)
sourcesIE = solveSoftIE(pulse_meshIE,
                sphericalWaveExcitation,
                wavenumber,
                distance_to_edge_tol,
                near_singular_tol)
pulse_meshIEnd = buildPulseMesh(mesh_filename, src_quadrature_rule, test_quadrature_rule)
sourcesIEnd = solveSoftIENormalDeriv(pulse_meshIEnd,
                sphericalWaveExcitationNormalDeriv,
                wavenumber)
exportSourcesGmsh(mesh_filename, "sourcesIE_nzcyl_sph_caps_real", real.(sourcesIE))
exportSourcesGmsh(mesh_filename, "sourcesIE_nzcyl_sph_caps_mag", abs.(sourcesIE))
exportSourcesGmsh(mesh_filename, "sourcesIEnd_nzcyl_sph_caps_real", real.(sourcesIEnd))
exportSourcesGmsh(mesh_filename, "sourcesIEnd_nzcyl_sph_caps_mag", abs.(sourcesIEnd))
exportSourcesGmsh(mesh_filename, "scale_factor_nzcyl_sph_caps_real", real.(sourcesIEnd ./ sourcesIE))
exportSourcesGmsh(mesh_filename, "scale_factor_nzcyl_sph_caps_mag", abs.(sourcesIEnd ./ sourcesIE))
exportSourcesGmsh(mesh_filename, "rel_error_nzcyl_sph_caps_real", real.((sourcesIE - sourcesIEnd) ./ sourcesIE))
exportSourcesGmsh(mesh_filename, "rel_error_nzcyl_sph_caps_mag", abs.((sourcesIE - sourcesIEnd) ./ sourcesIE))
l2norm_error = norm(sourcesIE .- sourcesIEnd) / norm(sourcesIE)
println("Q10 Error = ", l2norm_error)

println("\nQ10: How does the soft IE compare to the soft IE normal deriv when the mesh is a near zero height cylinder w/ spherical end caps oriented on y axis w/ spherical wave?")
mesh_filename = "examples/simple/cylinder_nearzero_with_spherical_caps_rot_1851.msh"
l, m = 0, 0
lambda=2.0
wavenumber = 2*pi/lambda + 0*im
sphericalWaveExcitation(x_test, y_test, z_test) = sphericalWave(excitation_amplitude, wavenumber, [x_test,y_test,z_test], l, m)
sphericalWaveExcitationNormalDeriv(x_test, y_test, z_test, normal) = sphericalWaveNormalDerivative(excitation_amplitude, wavenumber, [x_test,y_test,z_test], l, m, normal)
pulse_meshIE = buildPulseMesh(mesh_filename, src_quadrature_rule, test_quadrature_rule)
sourcesIE = solveSoftIE(pulse_meshIE,
                sphericalWaveExcitation,
                wavenumber,
                distance_to_edge_tol,
                near_singular_tol)
pulse_meshIEnd = buildPulseMesh(mesh_filename, src_quadrature_rule, test_quadrature_rule)
sourcesIEnd = solveSoftIENormalDeriv(pulse_meshIEnd,
                sphericalWaveExcitationNormalDeriv,
                wavenumber)
exportSourcesGmsh(mesh_filename, "sourcesIE_nzcyl_sph_caps_rot_real", real.(sourcesIE))
exportSourcesGmsh(mesh_filename, "sourcesIE_nzcyl_sph_caps_rot_mag", abs.(sourcesIE))
exportSourcesGmsh(mesh_filename, "sourcesIEnd_nzcyl_sph_caps_rot_real", real.(sourcesIEnd))
exportSourcesGmsh(mesh_filename, "sourcesIEnd_nzcyl_sph_caps_rot_mag", abs.(sourcesIEnd))
exportSourcesGmsh(mesh_filename, "scale_factor_nzcyl_sph_caps_rot_real", real.(sourcesIEnd ./ sourcesIE))
exportSourcesGmsh(mesh_filename, "scale_factor_nzcyl_sph_caps_rot_mag", abs.(sourcesIEnd ./ sourcesIE))
exportSourcesGmsh(mesh_filename, "rel_error_nzcyl_sph_caps_rot_real", real.((sourcesIE - sourcesIEnd) ./ sourcesIE))
exportSourcesGmsh(mesh_filename, "rel_error_nzcyl_sph_caps_rot_mag", abs.((sourcesIE - sourcesIEnd) ./ sourcesIE))
l2norm_error = norm(sourcesIE .- sourcesIEnd) / norm(sourcesIE)
println("Q10 Error = ", l2norm_error)

println("\nQ11: How does the soft IE compare to the soft IE normal deriv when the mesh is a sphere r=1 centered at x=1 w/ spherical wave?")
mesh_filename = "examples/simple/sphere/sphere_1m_offcenter_1405.msh"
l, m = 0, 0
lambda=2.0
wavenumber = 2*pi/lambda + 0*im
sphericalWaveExcitation(x_test, y_test, z_test) = sphericalWave(excitation_amplitude, wavenumber, [x_test,y_test,z_test], l, m)
sphericalWaveExcitationNormalDeriv(x_test, y_test, z_test, normal) = sphericalWaveNormalDerivative(excitation_amplitude, wavenumber, [x_test,y_test,z_test], l, m, normal)
pulse_meshIE = buildPulseMesh(mesh_filename, src_quadrature_rule, test_quadrature_rule)
sourcesIE = solveSoftIE(pulse_meshIE,
                sphericalWaveExcitation,
                wavenumber,
                distance_to_edge_tol,
                near_singular_tol)
pulse_meshIEnd = buildPulseMesh(mesh_filename, src_quadrature_rule, test_quadrature_rule)
sourcesIEnd = solveSoftIENormalDeriv(pulse_meshIEnd,
                sphericalWaveExcitationNormalDeriv,
                wavenumber)
exportSourcesGmsh(mesh_filename, "sourcesIE_sphere_1m_offcenter_real", real.(sourcesIE))
exportSourcesGmsh(mesh_filename, "sourcesIE_sphere_1m_offcenter_mag", abs.(sourcesIE))
exportSourcesGmsh(mesh_filename, "sourcesIEnd_sphere_1m_offcenter_real", real.(sourcesIEnd))
exportSourcesGmsh(mesh_filename, "sourcesIEnd_sphere_1m_offcenter_mag", abs.(sourcesIEnd))
exportSourcesGmsh(mesh_filename, "scale_factor_sphere_1m_offcenter_real", real.(sourcesIEnd ./ sourcesIE))
exportSourcesGmsh(mesh_filename, "scale_factor_sphere_1m_offcenter_mag", abs.(sourcesIEnd ./ sourcesIE))
exportSourcesGmsh(mesh_filename, "rel_error_sphere_1m_offcenter_real", real.((sourcesIE - sourcesIEnd) ./ sourcesIE))
exportSourcesGmsh(mesh_filename, "rel_error_sphere_1m_offcenter_mag", abs.((sourcesIE - sourcesIEnd) ./ sourcesIE))
l2norm_error = norm(sourcesIE .- sourcesIEnd) / norm(sourcesIE)
println("Q11 Error = ", l2norm_error)

println("\nQ12: How does the soft IE compare to the soft IE normal deriv when the mesh is a sphere r=2 centered at origin w/ spherical wave?")
mesh_filename = "examples/simple/sphere/sphere_2m_1681.msh"
l, m = 0, 0
lambda=3.7
wavenumber = 2*pi/lambda + 0*im
sphericalWaveExcitation(x_test, y_test, z_test) = sphericalWave(excitation_amplitude, wavenumber, [x_test,y_test,z_test], l, m)
sphericalWaveExcitationNormalDeriv(x_test, y_test, z_test, normal) = sphericalWaveNormalDerivative(excitation_amplitude, wavenumber, [x_test,y_test,z_test], l, m, normal)
pulse_meshIE = buildPulseMesh(mesh_filename, src_quadrature_rule, test_quadrature_rule)
sourcesIE = solveSoftIE(pulse_meshIE,
                sphericalWaveExcitation,
                wavenumber,
                distance_to_edge_tol,
                near_singular_tol)
pulse_meshIEnd = buildPulseMesh(mesh_filename, src_quadrature_rule, test_quadrature_rule)
sourcesIEnd = solveSoftIENormalDeriv(pulse_meshIEnd,
                sphericalWaveExcitationNormalDeriv,
                wavenumber)
exportSourcesGmsh(mesh_filename, "sourcesIE_sphere_2m_real", real.(sourcesIE))
exportSourcesGmsh(mesh_filename, "sourcesIE_sphere_2m_mag", abs.(sourcesIE))
exportSourcesGmsh(mesh_filename, "sourcesIEnd_sphere_2m_real", real.(sourcesIEnd))
exportSourcesGmsh(mesh_filename, "sourcesIEnd_sphere_2m_mag", abs.(sourcesIEnd))
exportSourcesGmsh(mesh_filename, "scale_factor_sphere_2m_real", real.(sourcesIEnd ./ sourcesIE))
exportSourcesGmsh(mesh_filename, "scale_factor_sphere_2m_mag", abs.(sourcesIEnd ./ sourcesIE))
exportSourcesGmsh(mesh_filename, "rel_error_sphere_2m_real", real.((sourcesIE - sourcesIEnd) ./ sourcesIE))
exportSourcesGmsh(mesh_filename, "rel_error_sphere_2m_mag", abs.((sourcesIE - sourcesIEnd) ./ sourcesIE))
l2norm_error = norm(sourcesIE .- sourcesIEnd) / norm(sourcesIE)
println("Q12 Error = ", l2norm_error)

println("\nQ13: How does the soft IE compare to the soft IE normal deriv when the mesh is a sphere r=2 centered at x=1 w/ spherical wave?")
mesh_filename = "examples/simple/sphere/sphere_2m_offcenter_1685.msh"
l, m = 0, 0
lambda=3.7
wavenumber = 2*pi/lambda + 0*im
sphericalWaveExcitation(x_test, y_test, z_test) = sphericalWave(excitation_amplitude, wavenumber, [x_test,y_test,z_test], l, m)
sphericalWaveExcitationNormalDeriv(x_test, y_test, z_test, normal) = sphericalWaveNormalDerivative(excitation_amplitude, wavenumber, [x_test,y_test,z_test], l, m, normal)
pulse_meshIE = buildPulseMesh(mesh_filename, src_quadrature_rule, test_quadrature_rule)
sourcesIE = solveSoftIE(pulse_meshIE,
                sphericalWaveExcitation,
                wavenumber,
                distance_to_edge_tol,
                near_singular_tol)
pulse_meshIEnd = buildPulseMesh(mesh_filename, src_quadrature_rule, test_quadrature_rule)
sourcesIEnd = solveSoftIENormalDeriv(pulse_meshIEnd,
                sphericalWaveExcitationNormalDeriv,
                wavenumber)
exportSourcesGmsh(mesh_filename, "sourcesIE_sphere_2m_offcenter_real", real.(sourcesIE))
exportSourcesGmsh(mesh_filename, "sourcesIE_sphere_2m_offcenter_mag", abs.(sourcesIE))
exportSourcesGmsh(mesh_filename, "sourcesIEnd_sphere_2m_offcenter_real", real.(sourcesIEnd))
exportSourcesGmsh(mesh_filename, "sourcesIEnd_sphere_2m_offcenter_mag", abs.(sourcesIEnd))
exportSourcesGmsh(mesh_filename, "scale_factor_sphere_2m_offcenter_real", real.(sourcesIEnd ./ sourcesIE))
exportSourcesGmsh(mesh_filename, "scale_factor_sphere_2m_offcenter_mag", abs.(sourcesIEnd ./ sourcesIE))
exportSourcesGmsh(mesh_filename, "rel_error_sphere_2m_offcenter_real", real.((sourcesIE - sourcesIEnd) ./ sourcesIE))
exportSourcesGmsh(mesh_filename, "rel_error_sphere_2m_offcenter_mag", abs.((sourcesIE - sourcesIEnd) ./ sourcesIE))
l2norm_error = norm(sourcesIE .- sourcesIEnd) / norm(sourcesIE)
println("Q13 Error = ", l2norm_error)

println("\nQ14: How does the soft IE compare to the soft IE normal deriv when the mesh is a half sphere r=1 centered at origin w/ spherical wave?")
mesh_filename = "examples/simple/sphere/half_sphere_1m_1765.msh"
l, m = 0, 0
lambda=2.2
wavenumber = 2*pi/lambda + 0*im
sphericalWaveExcitation(x_test, y_test, z_test) = sphericalWave(excitation_amplitude, wavenumber, [x_test,y_test,z_test], l, m)
sphericalWaveExcitationNormalDeriv(x_test, y_test, z_test, normal) = sphericalWaveNormalDerivative(excitation_amplitude, wavenumber, [x_test,y_test,z_test], l, m, normal)
pulse_meshIE = buildPulseMesh(mesh_filename, src_quadrature_rule, test_quadrature_rule)
sourcesIE = solveSoftIE(pulse_meshIE,
                sphericalWaveExcitation,
                wavenumber,
                distance_to_edge_tol,
                near_singular_tol)
pulse_meshIEnd = buildPulseMesh(mesh_filename, src_quadrature_rule, test_quadrature_rule)
sourcesIEnd = solveSoftIENormalDeriv(pulse_meshIEnd,
                sphericalWaveExcitationNormalDeriv,
                wavenumber)
exportSourcesGmsh(mesh_filename, "sourcesIE_half_sphere_1m_real", real.(sourcesIE))
exportSourcesGmsh(mesh_filename, "sourcesIE_half_sphere_1m_mag", abs.(sourcesIE))
exportSourcesGmsh(mesh_filename, "sourcesIEnd_half_sphere_1m_real", real.(sourcesIEnd))
exportSourcesGmsh(mesh_filename, "sourcesIEnd_half_sphere_1m_mag", abs.(sourcesIEnd))
exportSourcesGmsh(mesh_filename, "scale_factor_half_sphere_1m_real", real.(sourcesIEnd ./ sourcesIE))
exportSourcesGmsh(mesh_filename, "scale_factor_half_sphere_1m_mag", abs.(sourcesIEnd ./ sourcesIE))
exportSourcesGmsh(mesh_filename, "rel_error_half_sphere_1m_real", real.((sourcesIE - sourcesIEnd) ./ sourcesIE))
exportSourcesGmsh(mesh_filename, "rel_error_half_sphere_1m_mag", abs.((sourcesIE - sourcesIEnd) ./ sourcesIE))
l2norm_error = norm(sourcesIE .- sourcesIEnd) / norm(sourcesIE)
println("Q14 Error = ", l2norm_error)

# println("\nQ15: How does the soft IE compare to the soft IE normal deriv when the mesh is a sphere r=1 centered at origin made from scratch w/ spherical wave?")
# mesh_filename = "examples/simple/scratch_sphere_1m_1678.msh"
# l, m = 0, 0
# lambda=2.2
# wavenumber = 2*pi/lambda + 0*im
# sphericalWaveExcitation(x_test, y_test, z_test) = sphericalWave(excitation_amplitude, wavenumber, [x_test,y_test,z_test], l, m)
# sphericalWaveExcitationNormalDeriv(x_test, y_test, z_test, normal) = sphericalWaveNormalDerivative(excitation_amplitude, wavenumber, [x_test,y_test,z_test], l, m, normal)
# pulse_meshIE = buildPulseMesh(mesh_filename, src_quadrature_rule, test_quadrature_rule)
# sourcesIE = solveSoftIE(pulse_meshIE,
#                 sphericalWaveExcitation,
#                 wavenumber,
#                 distance_to_edge_tol,
#                 near_singular_tol)
# pulse_meshIEnd = buildPulseMesh(mesh_filename, src_quadrature_rule, test_quadrature_rule)
# sourcesIEnd = solveSoftIENormalDeriv(pulse_meshIEnd,
#                 sphericalWaveExcitationNormalDeriv,
#                 wavenumber)
# exportSourcesGmsh(mesh_filename, "sourcesIE_scratch_sphere_1m_real", real.(sourcesIE))
# exportSourcesGmsh(mesh_filename, "sourcesIE_scratch_sphere_1m_mag", abs.(sourcesIE))
# exportSourcesGmsh(mesh_filename, "sourcesIEnd_scratch_sphere_1m_real", real.(sourcesIEnd))
# exportSourcesGmsh(mesh_filename, "sourcesIEnd_scratch_sphere_1m_mag", abs.(sourcesIEnd))
# exportSourcesGmsh(mesh_filename, "scale_factor_scratch_sphere_1m_real", real.(sourcesIEnd ./ sourcesIE))
# exportSourcesGmsh(mesh_filename, "scale_factor_scratch_sphere_1m_mag", abs.(sourcesIEnd ./ sourcesIE))
# exportSourcesGmsh(mesh_filename, "rel_error_scratch_sphere_1m_real", real.((sourcesIE - sourcesIEnd) ./ sourcesIE))
# exportSourcesGmsh(mesh_filename, "rel_error_scratch_sphere_1m_mag", abs.((sourcesIE - sourcesIEnd) ./ sourcesIE))
# l2norm_error = norm(sourcesIE .- sourcesIEnd) / norm(sourcesIE)
# println("Q15 Error = ", l2norm_error)
