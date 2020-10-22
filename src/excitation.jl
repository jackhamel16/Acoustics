function planeWave(amplitude::Float64,
                   wavevector::Array{Complex{Float64}, 1},
                   position::Array{Float64, 1})
    amplitude * exp(-1*im*dot(wavevector, position))
end
