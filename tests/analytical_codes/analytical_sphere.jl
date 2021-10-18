# Dependencies: mesh.jl math.jl

################################################################################
# Thie file contains functions to compute the analytical solution to a sphere  #
# scatterer placed at the origin with an arbitrary incident spherical wave.    #
################################################################################

include("../../src/includes.jl")

function computeAnalyticalSolution(wavenumber, l, m, radius, mesh_filename)
    lm_idx = l + m + l^2 + 1
    pulse_mesh = buildPulseMesh(mesh_filename, gauss1rule, gauss1rule)
    sources_analytical = Array{Complex{Float64}, 1}(undef, pulse_mesh.num_elements)
    for element_idx in 1:pulse_mesh.num_elements
        position = computeCentroid(pulse_mesh.nodes[pulse_mesh.elements[element_idx,:],:])
        r = sqrt(sum(position.^2))
        theta, phi = acos(position[3] / r), atan(position[2], position[1])
        sources_analytical[element_idx] = -1im * wavenumber * sphericalHarmonics(theta, phi, l)[1][lm_idx] / ((wavenumber*radius)^2 * sphericalHankel2(l, real(wavenumber)*radius))
    end
    sources_analytical
end
