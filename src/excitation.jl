include("math.jl")

function planeWave(amplitude::Float64,
                   wavevector::Array{Complex{Float64}, 1},
                   position::Array{Float64, 1})
    amplitude * exp(-1*im*dot(wavevector, position))
end

function sphericalWave(amplitude::Float64,
                       wavenumber::Float64,
                       position::Array{Float64, 1},
                       l::Int64,
                       m::Int64)
    r = sqrt(sum(position.^2))
    theta, phi = acos(position[3] / r), atan(position[2], position[1])
    Ylm, dYlm_dtheta, dYlm_dphi, l_ind, m_ind = computeSpherHarms(theta, phi, l)
    lm_idx = l + m + l^2
    return(amplitude * Ylm[lm_idx] * sphericalBesselj(l, wavenumber*r))
end
