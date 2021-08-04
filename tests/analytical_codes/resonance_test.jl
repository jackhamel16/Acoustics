using DataFrames
using GLM
using Plots

include("../../src/includes.jl")
include("analytical_sphere.jl")

###############################################################################
# This test will run the sound-soft IE, normal derivative IE and CFIE for a 1266
# element sphere over a range of wavelengths. The results are compared with
# analytical solutions andthe l2-norm of the errors is plotted against
# wavelength.  The condition number of the z matrices is also plotted against
# wavelength.  The CFIE should not have spikes in condition number and error
# unlike the other IEs which spike at resonances.  The resonances correspond to
# when the RHS is zero in the method of moments implementations.

# determinant is commented since it is practically zero at all times due to the
# entries of the z matrix being on the order of 1e-7
###############################################################################

excitation_amplitude = 1.0
src_quadrature_rule = gauss7rule
test_quadrature_rule = gauss1rule
distance_to_edge_tol = 1e-12
near_singular_tol = 1.0
softIE_weight = 0.5
#set up sphere
radius = 1.0
l, m = 0, 0


# softIE modes occur when phi_inc is zero on sphere
#     l,m=0 modes: 2, 1, 2/3
num_freqs = 30
max_lambda = 11
min_lambda = 9
lambdas = [i for i in range(max_lambda,stop=min_lambda,length=num_freqs)]
num_elements = 1266
mesh_filename = string("examples/test/sphere_1m_",num_elements,".msh")
pulse_mesh = buildPulseMesh(mesh_filename, src_quadrature_rule, test_quadrature_rule)

l2errors_softIE = Array{Float64, 1}(undef, num_freqs)
l2errors_IENormalDeriv = Array{Float64, 1}(undef, num_freqs)
l2errors_CFIE = Array{Float64, 1}(undef, num_freqs)
cond_nums_softIE = Array{Float64, 1}(undef, num_freqs)
cond_nums_IENormalDeriv = Array{Float64, 1}(undef, num_freqs)
cond_nums_CFIE = Array{Float64, 1}(undef, num_freqs)
dets_softIE = Array{Float64, 1}(undef, num_freqs)
dets_IENormalDeriv = Array{Float64, 1}(undef, num_freqs)
dets_CFIE = Array{Float64, 1}(undef, num_freqs)

z_matrix_softIE_save, z_matrix_IENormalDeriv_save, z_matrix_CFIE_save = zeros(num_elements,num_elements), zeros(num_elements,num_elements), zeros(num_elements,num_elements)

for run_idx in 1:num_freqs
    lambda = lambdas[run_idx]
    println("Running lambda = ", lambda)
    wavenumber = 2*pi / lambda + 0*im
    sphericalWaveExcitation(x_test, y_test, z_test) = sphericalWave(excitation_amplitude, real(wavenumber), [x_test,y_test,z_test], l, m)
    sphericalWaveExcitationNormalDeriv(x_test, y_test, z_test, normal) = sphericalWaveNormalDerivative(excitation_amplitude, real(wavenumber), [x_test,y_test,z_test], l, m, normal)

    println("Running softIE")
    @time sources_softIE, z_matrix_softIE, rhs = solveSoftIE(pulse_mesh,
                    sphericalWaveExcitation,
                    wavenumber,
                    distance_to_edge_tol,
                    near_singular_tol,
                    true)

    real_filename = string("sources_real_softIE_sphere",num_elements)
    imag_filename = string("sources_imag_softIE_sphere",num_elements)
    mag_filename = string("sources_mag_softIE_sphere",num_elements)
    exportSourcesGmsh(mesh_filename, real_filename, real.(sources_softIE))
    exportSourcesGmsh(mesh_filename, imag_filename, imag.(sources_softIE))
    exportSourcesGmsh(mesh_filename, mag_filename, abs.(sources_softIE))

    println("Running softIENormalDeriv")
    @time sources_IENormalDeriv, z_matrix_IENormalDeriv = solveSoftIENormalDeriv(pulse_mesh,
                    sphericalWaveExcitationNormalDeriv,
                    wavenumber,
                    true)

    real_filename = string("sources_real_softIENormalDeriv_sphere",num_elements)
    imag_filename = string("sources_imag_softIENormalDeriv_sphere",num_elements)
    mag_filename = string("sources_mag_softIENormalDeriv_sphere",num_elements)
    exportSourcesGmsh(mesh_filename, real_filename, real.(sources_IENormalDeriv))
    exportSourcesGmsh(mesh_filename, imag_filename, imag.(sources_IENormalDeriv))
    exportSourcesGmsh(mesh_filename, mag_filename, abs.(sources_IENormalDeriv))

    println("Running softCFIE")
    @time sources_CFIE, z_matrix_CFIE = solveSoftCFIE(pulse_mesh,
                    sphericalWaveExcitation,
                    sphericalWaveExcitationNormalDeriv,
                    wavenumber,
                    distance_to_edge_tol,
                    near_singular_tol,
                    softIE_weight,
                    true)

    real_filename = string("sources_real_softCFIE_sphere",num_elements)
    imag_filename = string("sources_imag_softCFIE_sphere",num_elements)
    mag_filename = string("sources_mag_softCFIE_sphere",num_elements)
    exportSourcesGmsh(mesh_filename, real_filename, real.(sources_CFIE))
    exportSourcesGmsh(mesh_filename, imag_filename, imag.(sources_CFIE))
    exportSourcesGmsh(mesh_filename, mag_filename, abs.(sources_CFIE))

    sources_analytical = computeAnalyticalSolution(wavenumber, radius, mesh_filename)

    l2errors_softIE[run_idx] = sqrt(sum(abs.(sources_analytical .- sources_softIE).^2)/sum(abs.(sources_analytical).^2))
    l2errors_IENormalDeriv[run_idx] = sqrt(sum(abs.(sources_analytical .- sources_IENormalDeriv).^2)/sum(abs.(sources_analytical).^2))
    l2errors_CFIE[run_idx] = sqrt(sum(abs.(sources_analytical .- sources_CFIE).^2)/sum(abs.(sources_analytical).^2))

    cond_nums_softIE[run_idx] = cond(z_matrix_softIE)
    cond_nums_IENormalDeriv[run_idx] = cond(z_matrix_IENormalDeriv)
    cond_nums_CFIE[run_idx] = cond(z_matrix_CFIE)

    # dets_softIE[run_idx] = det(z_matrix_softIE)
    # dets_IENormalDeriv[run_idx] = det(z_matrix_IENormalDeriv)
    # dets_CFIE[run_idx] = det(z_matrix_CFIE)
end

Plots.plot(lambdas, [l2errors_softIE,l2errors_IENormalDeriv,l2errors_CFIE],
           label=["Soft IE" "Soft IE deriv" "Soft CFIE"],
           title="Sources vs Aanlytic Results on Sphere r=1",
           xlabel="Wavelength",
           ylabel="l2-Error",
           yaxis=:log)
savefig("resonance_test_l2errors_plot")

Plots.plot(lambdas, [cond_nums_softIE,cond_nums_IENormalDeriv,cond_nums_CFIE],
           label=["Soft IE" "Soft IE deriv" "Soft CFIE"],
           title="Condition Number of Z Matrix of Sphere r=1",
           xlabel="Wavelength",
           ylabel="Condition Number",
           yaxis=:log)
savefig("resonance_test_condition_numbers_plot")

# Plots.plot(lambdas, [dets_softIE,dets_IENormalDeriv,dets_CFIE],
#            label=["Soft IE" "Soft IE deriv" "Soft CFIE"],
#            title="Determinant of Z Matrix of Sphere r=1",
#            xlabel="Wavelength",
#            ylabel="Determinant")
# savefig("resonance_test_determinants_plot")

l2errors_file = open("resonance_test_l2errors.txt", "w")
l2errors_data = hcat(lambdas, l2errors_softIE, l2errors_IENormalDeriv, l2errors_CFIE)
println(l2errors_file, "[lambda, l2-error soft IE, l2-error soft IE ND, l2-error soft CFIE]")
cond_nums_file = open("resonance_test_cond_nums.txt", "w")
cond_nums_data = hcat(lambdas, cond_nums_softIE, cond_nums_IENormalDeriv, cond_nums_CFIE)
println(cond_nums_file, "[lambda, l2-error soft IE, l2-error soft IE ND, l2-error soft CFIE]")
# dets_file = open("resonance_test_dets.txt", "w")
# dets_data = hcat(lambdas, dets_softIE, dets_IENormalDeriv, dets_CFIE)
# println(dets_file, "[lambda, l2-error soft IE, l2-error soft IE ND, l2-error soft CFIE]")
for run_idx in 1:num_freqs
    println(l2errors_file, l2errors_data[run_idx,:])
    println(cond_nums_file, cond_nums_data[run_idx,:])
    # println(dets_file, dets_data[run_idx,:])
end
close(l2errors_file)
close(cond_nums_file)
# close(dets_file)
