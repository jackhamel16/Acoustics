using DataFrames
using GLM
using Plots
using Printf

include("../../src/includes.jl")
include("analytical_sphere.jl")

excitation_amplitude = 1.0
# lambda = 10.0
# wavenumber = 2*pi / lambda + 0*im
src_quadrature_rule = gauss7rule
test_quadrature_rule = gauss1rule
distance_to_edge_tol = 1e-12
near_singular_tol = 1.0
#set up sphere
radius = 1.0
l, m = 0, 0
softIE_weight = 0.5

num_levels = 3
compression_distance = 1.5
ACA_approximation_tol = 1e-6
gmres_tol = 1e-12

# note: make sure lowest lambda does not encrouch on resonances
num_elements = [1266, 3788, 8010, 19034]#, 39610, 70118]

edges_per_wavelength = 350 # hold ratio of lambda to edge length constant
sphere_area = pi * radius^2
element_areas = sphere_area ./ num_elements
avg_element_edge_lengths = sqrt.(element_areas .* 4 ./ sqrt(3)) #assumes equilateral triangle
lambdas = edges_per_wavelength .* avg_element_edge_lengths
wavenumbers = 2 .* pi ./ lambdas .+ 0*im

l2errors = Array{Float64, 1}(undef, 0)
all_metrics = Array{ACAMetrics, 1}(undef, length(num_elements))
all_times = Array{Float64, 1}(undef, length(num_elements))


for run_idx in 1:length(num_elements)
    println("Running ", num_elements[run_idx], " Unknowns")
    sphericalWaveExcitation(x_test, y_test, z_test) = sphericalWave(excitation_amplitude, real(wavenumbers[run_idx]), [x_test,y_test,z_test], l, m)
    sphericalWaveNormalDerivExcitation(x_test, y_test, z_test, normal) = sphericalWaveNormalDerivative(excitation_amplitude, real(wavenumbers[run_idx]), [x_test,y_test,z_test], l, m, normal)
    mesh_filename = string("examples/test/spheres/sphere_1m_",num_elements[run_idx],".msh")
    pulse_mesh = buildPulseMesh(mesh_filename, src_quadrature_rule, test_quadrature_rule)

    run_time = @elapsed sources, octree, metrics = solveSoftCFIEACA(pulse_mesh,
                                        num_levels,
                                        sphericalWaveExcitation,
                                        sphericalWaveNormalDerivExcitation,
                                        wavenumbers[run_idx],
                                        softIE_weight,
                                        distance_to_edge_tol,
                                        near_singular_tol,
                                        compression_distance,
                                        ACA_approximation_tol)

    real_filename = string("sources_real_softCFIEACA_sphere",num_elements[run_idx])
    imag_filename = string("sources_imag_softCFIEACA_sphere",num_elements[run_idx])
    mag_filename = string("sources_mag_softCFIEACA_sphere",num_elements[run_idx])
    exportSourcesGmsh(mesh_filename, real_filename, real.(sources))
    exportSourcesGmsh(mesh_filename, imag_filename, imag.(sources))
    exportSourcesGmsh(mesh_filename, mag_filename, abs.(sources))

    sources_analytical = computeAnalyticalSolution(wavenumbers[run_idx], l, m, radius, mesh_filename)

    append!(l2errors, sqrt(sum(abs.(sources_analytical .- sources).^2)/sum(abs.(sources_analytical).^2)))
    all_metrics[run_idx] = metrics
    all_times[run_idx] = run_time
    println("Run time = ", run_time, " seconds")
    printACAMetrics(metrics)
end

# Fit line to log of error data
error_data = DataFrame(X=log.(sqrt.(num_elements)), Y=log.(l2errors))
error_linreg_output = lm(@formula(Y ~ X), error_data)

intercept = coef(error_linreg_output)[1]
error_slope = coef(error_linreg_output)[2]
error_fit(n) = exp(intercept + error_slope*log(n))
n = [i for i in 1000:100:num_elements[length(num_elements)]*1.05]
predicted_errors = error_fit.(sqrt.(n))

error_labels = [@sprintf("%.2E",l2errors[i]) for i in 1:length(l2errors)]
xtick_labels=string.(round.(sqrt.(num_elements),digits=1))
plot(sqrt.(n), predicted_errors, title="1m Sphere ACA vs Analytical Solution", label="Sound-Soft", xlabel="sqrt(N), N is number of unknowns", ylabel="l2-Error", xticks=(sqrt.(num_elements),xtick_labels), yticks=(l2errors,error_labels))
Plots.scatter!(sqrt.(num_elements), l2errors, label="", xaxis=:log, yaxis=:log, size=(800,600))
savefig("sphere_convergence_results_softCFIEACA")
println("Convergence rate = ", error_slope)

# Fit line to log of time data
time_data = DataFrame(X=log.(sqrt.(num_elements)), Y=log.(all_times))
time_linreg_output = lm(@formula(Y ~ X), time_data)

intercept = coef(time_linreg_output)[1]
time_slope = coef(time_linreg_output)[2]
time_fit(n) = exp(intercept + time_slope*log(n))
n = [i for i in 1000:100:num_elements[length(num_elements)]*1.05]
predicted_times = time_fit.(sqrt.(n))

time_labels = [@sprintf("%.2E",all_times[i]) for i in 1:length(all_times)]
plot(sqrt.(n), predicted_times, title="1m Sphere ACA Runtimes", label="Sound-Soft", xlabel="sqrt(N), N is number of unknowns", ylabel="Seconds", xticks=(sqrt.(num_elements),xtick_labels), yticks=(all_times,time_labels))
Plots.scatter!(sqrt.(num_elements), all_times, label="", xaxis=:log, yaxis=:log, size=(800,600))
savefig("sphere_runtime_results_softCFIEACA")
println("Run Time Slope = ", time_slope)

#Check if convergence rate is correct
convergence_rates = [-2.3787216946889758, -2.6944277683731173, 2.863990233302726] # using 2, 3, or 4 meshes, 7pnt src 1 pnt test
expected_convergence_rate = convergence_rates[size(num_elements)[1]-1]
convergence_error = abs((expected_convergence_rate - error_slope)/expected_convergence_rate)
tolerance = 1e-6
if convergence_error > tolerance
    println("TEST FAILED:")
    println("Convergence rate not within ", tolerance, " of expected rate of ", expected_convergence_rate)
else
    println("TEST PASSED")
end

#output data
output_file = open("sphere_convergence_data_softCFIEACA.txt", "w")
output_data = hcat(num_elements, l2errors)
println(output_file, "Convergence rate = ", error_slope)
println(output_file, "[num unknowns, l2-error]")
for run_idx in 1:length(num_elements)
    println(output_file, output_data[run_idx,:])
end
println(output_file, "\n")
for run_idx in 1:length(num_elements)
    println(output_file, "Run Number: ", run_idx)
    println(output_file, "Run time = ", all_times[run_idx], " seconds")
    printACAMetrics(all_metrics[run_idx], output_file)
end
close(output_file)
