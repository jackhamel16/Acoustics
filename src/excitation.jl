# dependencies: math.jl

function planeWave(amplitude,
                   wavevector::AbstractArray{T, 1},
                   position::Array{Float64, 1}) where T
    amplitude * exp(-im*dot(position, wavevector))
end

function planeWaveNormalDerivative(amplitude,
                                   wavevector::AbstractArray{T, 1},
                                   position::Array{Float64, 1},
                                   normal::Array{Float64, 1}) where T
    -im * dot(normal, wavevector) * planeWave(amplitude, wavevector, position)
end

function sphericalWave(amplitude,
                       wavenumber,
                       position::Array{Float64, 1},
                       l::Int64,
                       m::Int64)
    r = norm(position)
    theta, phi = acos(position[3] / r), atan(position[2], position[1])
    Ylm, dYlm_dtheta, dYlm_dphi, l_ind, m_ind = sphericalHarmonics(theta, phi, l)
    lm_idx = l + m + l^2 + 1
    return(amplitude * Ylm[lm_idx] * sphericalBesselj(l, wavenumber*r))
end

@views function sphericalWaveNormalDerivative(amplitude,
                                              wavenumber,
                                              position::Array{Float64, 1},
                                              l::Int64,
                                              m::Int64,
                                              normal::Array{Float64, 1})
    x, y, z = position
    r = norm(position)
    theta, phi = acos(z / r), atan(y, x)
    lm_idx = l + m + l^2 + 1
    Ylm, dYlm_dtheta, dYlm_dphi, l_ind, m_ind = sphericalHarmonics(theta, phi, l)
    jlkr_over_r = sphericalBesselj(l,wavenumber*r)/r
    gradient_spherical_wave = [Ylm[lm_idx] * (l*jlkr_over_r - wavenumber * sphericalBesselj(l+1,wavenumber*r)),
                               jlkr_over_r * dYlm_dtheta[lm_idx],
                               jlkr_over_r * csc(theta) * dYlm_dphi[lm_idx]]
    r_0z = sqrt(x^2+y^2)
    transform_matrix = zeros(3, 3) # spherical vector -> cartesian vector
    transform_matrix[:,1] = position ./ r
    transform_matrix[:,2] = [x*z, y*z, -x^2-y^2] ./ (r*r_0z)
    transform_matrix[:,3] = [-y, x, 0] ./ r_0z
    gradient_spherical_wave_cartesian = transform_matrix * gradient_spherical_wave
    return(amplitude * dot(normal, transform_matrix * gradient_spherical_wave))
end

@views function sphericalWaveKDerivative(wavenumber,
                                         position::Array{Float64, 1},
                                         l::Int64,
                                         m::Int64)
    x, y, z = position
    r = norm(position)
    theta, phi = acos(z / r), atan(y, x)
    lm_idx = l + m + l^2 + 1
    Ylm, dYlm_dtheta, dYlm_dphi, l_ind, m_ind = sphericalHarmonics(theta, phi, l)
    kr = wavenumber * r
    return(2 * Ylm[lm_idx] * ((l+1)*sphericalBesselj(l, kr) - kr*sphericalBesselj(l+1, kr)))
end
