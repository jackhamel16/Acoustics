using GSL
using LinearAlgebra
using SpecialFunctions

function convertSpherical2Cartesian(spherical_coords::AbstractArray{T, 1}) where T
    x = spherical_coords[1]*sin(spherical_coords[2])*cos(spherical_coords[3])
	y = spherical_coords[1]*sin(spherical_coords[2])*sin(spherical_coords[3])
	z = spherical_coords[1]*cos(spherical_coords[2])
	return([x, y, z])
end

function sphericalHarmonics(theta::Float64, phi::Float64, lmax::Int64)

    # Obtain the legendre polynomial and its array.
    # This function gives legendre polynomial for  0 <= l <= lmax,  and  0 <= m <= l

	cos_theta = cos(theta);

    (Leg, dLeg) = sf_legendre_deriv_array(GSL_SF_LEGENDRE_SPHARM, lmax, cos_theta);

    l_ind = Array{Int}(undef,lmax*(lmax+2)+1);
    m_ind = Array{Int}(undef,lmax*(lmax+2)+1);
    l_ind[1], m_ind[1] = 0, 0
    for ll = 1:lmax
        l_ind[(ll-1)*(ll+1)+2 : (ll)*(ll+2)+1] .= ll;
        m_ind[(ll-1)*(ll+1)+2 : (ll)*(ll+2)+1] = [-ll:1:ll;];
    end
    Index_numbers = convert(Array{Int}, 0.5.*(l_ind .+ 1).*(l_ind) .+ 1 .+ abs.(m_ind));
    Leg_lm = Leg[Index_numbers];
    dLeg_lm = dLeg[Index_numbers];

    # add (-1)^m multiplier for m > 0
    Leg_lm[(m_ind .> 0) .& (abs.(m_ind.%2) .> 0) ] .= Leg_lm[ (m_ind .> 0) .& (abs.(m_ind.%2) .> 0) ].*-1;
    dLeg_lm[(m_ind .> 0) .& (abs.(m_ind.%2) .> 0) ] .= dLeg_lm[ (m_ind .> 0) .& (abs.(m_ind.%2) .> 0) ].*-1;

    Ylm::Array{ComplexF64,1} = Leg_lm.*exp.(1im.*m_ind.*phi);
    dYlm_dtheta::Array{ComplexF64,1} = -sin(theta).*dLeg_lm.*exp.(1im.*m_ind.*phi);
    dYlm_dphi::Array{ComplexF64,1} = 1im.*m_ind.*Leg_lm.*exp.(1im.*m_ind.*phi);

    return (Ylm, dYlm_dtheta, dYlm_dphi, l_ind, m_ind);
end

function sphericalBesselj(n::Real, x)
    return(sqrt(pi / (2 * x)) * besselj(n+0.5, x))
end

function sphericalBessely(n::Real, x)
    return(sqrt(pi / (2 * x)) * bessely(n+0.5, x))
end

function sphericalHankel2(n::Real, x)
    return(sphericalBesselj(n, x) - im * sphericalBessely(n, x))
end

function sphericalHankel1(n::Real, x)
    return(sphericalBesselj(n, x) + im * sphericalBessely(n, x))
end
