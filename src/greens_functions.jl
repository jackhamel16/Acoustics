using LinearAlgebra

function scalarGreens(R::Float64, k::Complex{Float64})
    exp(-im*k*abs(R))/(4*pi*abs(R))
end

function singularScalarGreens(d::Float64,
                                P0_hat::Array{Float64, 2},
                                u_hat::Array{Float64, 2},
                                P0::Array{Float64, 1},
                                R0::Array{Float64, 1},
                                R_plus::Array{Float64, 1},
                                R_minus::Array{Float64, 1},
                                l_plus::Array{Float64, 1},
                                l_minus::Array{Float64, 1})
    integral = 0.0
    for i in 1:3 #i indicates triangle edge index
        integral += dot(P0_hat[i,:], u_hat[i,:]) *
                    (P0[i] * log((R_plus[i]+l_plus[i])/(R_minus[i]+l_minus[i]))-
                     abs(d) * (atan(P0[i]*l_plus[i], R0[i]^2+abs(d)*R_plus[i]) -
                               atan(P0[i]*l_minus[i], R0[i]^2+abs(d)*R_minus[i])))
    end
    integral
end

# function computeSingularParameters()
