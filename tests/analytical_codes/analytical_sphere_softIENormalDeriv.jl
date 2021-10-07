using DataFrames
using GLM
using Plots

include("../../src/includes.jl")
include("analytical_sphere.jl")

excitation_amplitude = 1.0
lambda = 6.0
wavenumber = 2*pi / lambda + 0*im
src_quadrature_rule = gauss7rule
test_quadrature_rule = gauss1rule
distance_to_edge_tol = 1e-12
near_singular_tol = 1.0
#set up sphere
radius = 1.0
l, m = 2, 1

num_elements = [1266]#, 3788]#, 8010]#, 19034]
l2errors = Array{Float64, 1}(undef, 0)

sphericalWaveExcitationNormalDeriv(x_test, y_test, z_test, normal) = sphericalWaveNormalDerivative(excitation_amplitude, real(wavenumber), [x_test,y_test,z_test], l, m, normal)
for run_idx in 1:length(num_elements)
    println("Running ", num_elements[run_idx], " Unknowns")
    mesh_filename = string("examples/test/spheres/sphere_1m_",num_elements[run_idx],".msh")
    pulse_mesh = buildPulseMesh(mesh_filename, src_quadrature_rule, test_quadrature_rule)

    @time sources = solveSoftIENormalDeriv(pulse_mesh,
                    sphericalWaveExcitationNormalDeriv,
                    wavenumber)

    real_filename = string("sources_real_softIENormalDeriv_l",l,"_m",m,"_sphere",num_elements[run_idx])
    imag_filename = string("sources_imag_softIENormalDeriv_l",l,"_m",m,"_sphere",num_elements[run_idx])
    mag_filename = string("sources_mag_softIENormalDeriv_l",l,"_m",m,"_sphere",num_elements[run_idx])
    exportSourcesGmsh(mesh_filename, real_filename, real.(sources))
    exportSourcesGmsh(mesh_filename, imag_filename, imag.(sources))
    exportSourcesGmsh(mesh_filename, mag_filename, abs.(sources))

    sources_analytical = computeAnalyticalSolution(wavenumber, l, m, radius, mesh_filename)
    real_analytical_filename = string("sources_real_analytical_l",l,"_m",m,"_sphere",num_elements[run_idx])
    imag_analytical_filename = string("sources_imag_analytical_l",l,"_m",m,"_sphere",num_elements[run_idx])
    mag_analytical_filename = string("sources_mag_analytical_l",l,"_m",m,"_sphere",num_elements[run_idx])
    exportSourcesGmsh(mesh_filename, real_analytical_filename, real.(sources_analytical))
    exportSourcesGmsh(mesh_filename, imag_analytical_filename, imag.(sources_analytical))
    exportSourcesGmsh(mesh_filename, mag_analytical_filename, abs.(sources_analytical))

    scale_factors = sources_analytical ./ sources
    exportSourcesGmsh(mesh_filename, string("scale_factors_real_l",l,"_m",m,"_sphere"), real.(scale_factors))
    exportSourcesGmsh(mesh_filename, string("scale_factors_imag_l",l,"_m",m,"_sphere"), imag.(scale_factors))
    exportSourcesGmsh(mesh_filename, string("scale_factors_mag_l",l,"_m",m,"_sphere"), abs.(scale_factors))
    println("Avg scale factor = ", sum(scale_factors) / pulse_mesh.num_elements)
    println("Avg scale factor mag = ", sum(abs.(sources_analytical) ./ abs.(sources)) / pulse_mesh.num_elements)

    append!(l2errors, sqrt(sum(abs.(sources_analytical .- sources).^2)/sum(abs.(sources_analytical).^2)))
end

# Fit line to log of data
error_data = DataFrame(X=log.(sqrt.(num_elements)), Y=log.(l2errors))
linreg_output = lm(@formula(Y ~ X), error_data)

intercept = coef(linreg_output)[1]
slope = coef(linreg_output)[2]
error_fit(n) = exp(intercept + slope*log(n))

n = [i for i in 1000:100:20000]
predicted_errors = error_fit.(sqrt.(n))

Plots.scatter(sqrt.(num_elements), l2errors, label="", xaxis=:log, yaxis=:log, size=(800,600))
plot!(sqrt.(n), predicted_errors, title="1m Sphere vs Analytical Solution", label="Sound-Soft Normal Deriv", xlabel="sqrt(N), N is number of unknowns", ylabel="l2-Error")
savefig(string("sphere_convergence_results_softIENormalDeriv_l",l,"_m",m))
println("Convergence rate = ", slope)

#Check if convergence rate is correct
convergence_rates = [-1.1579399752858406, -1.1200508751294938, -1.0940611094829429] # using 2, 3, or 4 meshes, 7pnt src 1 pnt test, 10 lambda
expected_convergence_rate = convergence_rates[size(num_elements)[1]-1]
convergence_error = abs((expected_convergence_rate - slope)/expected_convergence_rate)
tolerance = 1e-6
if convergence_error > tolerance
    println("TEST FAILED:")
    println("Convergence rate not within ", tolerance, " of expected rate of ", expected_convergence_rate)
else
    println("TEST PASSED")
end

#output data
output_file = open(string("sphere_convergence_data_softIENormalDeriv_l",l,"_m",m,".txt"), "w")
output_data = hcat(num_elements, l2errors)
println(output_file, "Convergence rate = ", slope)
println(output_file, "[num unknowns, l2-error]")
for run_idx in 1:length(num_elements)
    println(output_file, output_data[run_idx,:])
end
close(output_file)
