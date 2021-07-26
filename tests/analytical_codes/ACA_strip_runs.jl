using DataFrames
using GLM
using Plots
using Printf

include("../../src/includes.jl")

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

num_levels = 4
compression_distance = 1.5
ACA_approximation_tol = 1e-4
gmres_tol = 1e-12

# note: make sure lowest lambda does not encrouch on resonances
num_elements = [640, 2560, 10240, 40960]#, 163840]

edges_per_wavelength = 100 # hold ratio of lambda to edge length constant
sphere_area = pi * radius^2
element_areas = sphere_area ./ num_elements
avg_element_edge_lengths = sqrt.(element_areas .* 4 ./ sqrt(3)) #assumes equilateral triangle
lambdas = edges_per_wavelength .* avg_element_edge_lengths
wavenumbers = 2 .* pi ./ lambdas .+ 0*im

all_metrics = Array{ACAMetrics, 1}(undef, length(num_elements))
all_times = Array{Float64, 1}(undef, length(num_elements))

for run_idx in 1:length(num_elements)
    println("Running ", num_elements[run_idx], " Unknowns")
    sphericalWaveExcitation(x_test, y_test, z_test) = sphericalWave(excitation_amplitude, real(wavenumbers[run_idx]), [x_test,y_test,z_test], l, m)
    mesh_filename = string("examples/simple/plates/rectangular_strip_",num_elements[run_idx],".msh")
    pulse_mesh = buildPulseMesh(mesh_filename, src_quadrature_rule, test_quadrature_rule)

    run_time = @elapsed sources, octree, metrics = solveSoundSoftIEACA(pulse_mesh,
                                        num_levels,
                                        sphericalWaveExcitation,
                                        wavenumbers[run_idx],
                                        distance_to_edge_tol,
                                        near_singular_tol,
                                        compression_distance,
                                        ACA_approximation_tol)

    real_filename = string("sources_real_softIEACA_strip",num_elements[run_idx])
    imag_filename = string("sources_imag_softIEACA_strip",num_elements[run_idx])
    mag_filename = string("sources_mag_softIEACA_strip",num_elements[run_idx])
    exportSourcesGmsh(mesh_filename, real_filename, real.(sources))
    exportSourcesGmsh(mesh_filename, imag_filename, imag.(sources))
    exportSourcesGmsh(mesh_filename, mag_filename, abs.(sources))

    all_metrics[run_idx] = metrics
    all_times[run_idx] = run_time
    println("Run time = ", run_time, " seconds")
    printACAMetrics(metrics)
end

# Fit line to log of time data
time_data = DataFrame(X=log.(sqrt.(num_elements)), Y=log.(all_times))
time_linreg_output = lm(@formula(Y ~ X), time_data)

intercept = coef(time_linreg_output)[1]
time_slope = coef(time_linreg_output)[2]
time_fit(n) = exp(intercept + time_slope*log(n))
n = [i for i in num_elements[1]*0.9:100:num_elements[length(num_elements)]*1.05]
predicted_times = time_fit.(sqrt.(n))

time_labels = [@sprintf("%.2E",all_times[i]) for i in 1:length(all_times)]
xtick_labels=string.(round.(sqrt.(num_elements),digits=1))
plot(sqrt.(n), predicted_times, title="5m Strip ACA Runtimes", label="Sound-Soft", xlabel="sqrt(N), N is number of unknowns", ylabel="Seconds", xticks=(sqrt.(num_elements),xtick_labels), yticks=(all_times,time_labels))
Plots.scatter!(sqrt.(num_elements), all_times, label="", xaxis=:log, yaxis=:log, size=(800,600))
savefig("strip_runtime_results_softIEACA")
println("Run Time Slope = ", time_slope)

#output data
output_file = open("strip_convergence_data_softIEACA.txt", "w")
println(output_file, "num unknowns")
for run_idx in 1:length(num_elements)
    println(output_file, num_elements[run_idx])
end
println(output_file, "\n")
for run_idx in 1:length(num_elements)
    println(output_file, "Run Number: ", run_idx)
    println(output_file, "Run time = ", all_times[run_idx], " seconds")
    printACAMetrics(all_metrics[run_idx], output_file)
end
close(output_file)
