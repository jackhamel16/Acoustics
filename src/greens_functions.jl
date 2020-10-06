using LinearAlgebra

# struct ScalarGreensSingularityIntegralParams
#     d::Float64,
#     P0_hat::Array{Float64, 2}
#     u_hat::Array{Float64, 2}
#     P0::Array{Float64, 1}
#     R0::Array{Float64, 1}
#     R_plus::Array{Float64, 1}
#     R_minus::Array{Float64, 1}
#     l_plus::Array{Float64, 1}
#     l_minus::Array{Float64, 1}
# end

function scalarGreens(R::Float64, k::Complex{Float64})
    exp(-im*k*abs(R))/(4*pi*abs(R))
end

function scalarGreensNonSingular(R::Float64, k::Complex{Float64})
    (exp(-im*k*abs(R))-1)/(4*pi*abs(R))
end

function scalarGreensIntegration(wavenumber::Complex{Float64},
                                 centroid_src::Array{Float64, 1},
                                 r_test::Array{Float64, 1},
                                 nodes::Array{Float64, 2},
                                 quadrature_rule::Array{Float64, 2},
                                 distance_to_edge_tol::Float64,
                                 near_singular_tol::Float64,
                                 is_singular::Bool)
    # This function encapsulates all possible integration routines for the
    # scalar Green's function over a source triangle.
    # distance_to_edge_tol is distance at which the projection of r_test must be
    #    to an edge or extension to ignore that edge's contribution to the integral
    # near_singular_tol is the number of max edge lengths away r_test can be
    #    before doing purely numerical integration
    max_edge_length = 0.0
    for edge_idx in 1:3
        edge_length = norm(nodes[edge_idx,:]-nodes[edge_idx%3+1,:])
        if edge_length > max_edge_length
            max_edge_length = edge_length
        end
    end
    if is_singular == true
        scalarGreensSingularIntegral(wavenumber, r_test, nodes, quadrature_rule,
                                     distance_to_edge_tol)
    elseif norm(r_test-centroid_src) > near_singular_tol*max_edge_length
        scalar_greens_integrand(x,y,z) = scalarGreens(norm([x,y,z]-r_test), wavenumber)
        integrateTriangle(nodes, scalar_greens_integrand, quadrature_rule[:,1:3],
                          quadrature_rule[:,4])
    else
        scalarGreensNearSingularIntegral(wavenumber, r_test, nodes,
                                         quadrature_rule, distance_to_edge_tol)
    end
end

function scalarGreensSingularityIntegral(r_test::Array{Float64, 1},
                                         nodes::Array{Float64, 2},
                                         epsilon::Float64)
    # Analytical integral over a triangle of 1/R
    # Source: Potential Integrals for Uniform and Linear Source
    #         Distributions on Polygonal and Polyhedral Domains
    #         by Wilton, Rao, Glisson , etc.
    #             See equation 5
    # Inputs are the outputs of computeScalarGreensSingularityIntegralParameters()
    # r_test is a 3 element array describing test points
    # nodes is a 3 x 3 array with each row a node of the source triangle
    # epsilon is the ignore contribution tolerance of distance of r_test projection to an edge
    d, P0_hat, u_hat, P0, R0, R_plus, R_minus, l_plus, l_minus =
        computeScalarGreensSingularityIntegralParameters(r_test, nodes)
    integral = 0.0
    for i in 1:3 #i indicates triangle edge index
        if P0[i] < epsilon # Indicates projection of r_test is on edge i or its
            continue  # extension which makes this contribution zero so skip it
        end
        integral += dot(P0_hat[i,:], u_hat[i,:]) *
                    (P0[i] * log((R_plus[i]+l_plus[i])/(R_minus[i]+l_minus[i]))-
                     abs(d) * (atan(P0[i]*l_plus[i], R0[i]^2+abs(d)*R_plus[i]) -
                               atan(P0[i]*l_minus[i], R0[i]^2+abs(d)*R_minus[i])))
    end
    integral / (4*pi)
end

function scalarGreensNearSingularIntegral(wavenumber::Complex{Float64},
                                          r_test::Array{Float64, 1},
                                          nodes::Array{Float64, 2},
                                          quadrature_rule::Array{Float64, 2},
                                          distance_to_edge_tol::Float64)
    # Used when r_test is close to source triangle, but not on it. Uses a
    # combination of analytical integration for the singular term and numerical
    # for the rest.
    non_singular_integrand(x,y,z) = scalarGreensNonSingular(norm([x,y,z]-r_test),
                                                            wavenumber)
    non_singular_integral = integrateTriangle(nodes, non_singular_integrand,
                                              quadrature_rule[:,1:3],
                                              quadrature_rule[:,4])
    singular_integral = scalarGreensSingularityIntegral(r_test, nodes,
                                                        distance_to_edge_tol)
    singular_integral + non_singular_integral
end

function scalarGreensSingularIntegral(wavenumber::Complex{Float64},
                                      r_test::Array{Float64, 1},
                                      nodes::Array{Float64, 2},
                                      quadrature_rule::Array{Float64, 2},
                                      distance_to_edge_tol::Float64)
    # Computes the integral of the scalar greens function for self-interactions
    # i.e. r_test is in the source triangle described by nodes
    total_integral = 0.0
    for triangle_idx in 1:3
        sub_nodes = copy(nodes)
        sub_nodes[triangle_idx,:] = r_test
        total_integral += scalarGreensNearSingularIntegral(wavenumber, r_test,
                                                           sub_nodes, quadrature_rule,
                                                           distance_to_edge_tol)
    end
    total_integral
end

function computeScalarGreensSingularityIntegralParameters(r_test::Array{Float64, 1},
                                                          nodes::Array{Float64, 2})
    # Computes the input parameters for scalarGreensSingularityIntegral()
    # r_test is the observation point
    # nodes are the three triangle nodes
    r_minus = nodes # beginning points of edges
    r_plus = circshift(nodes, (-1,0)) # ending points of edges

    normal_non_unit = cross(r_plus[1,:]-r_minus[1,:], r_minus[3,:]-r_plus[3,:])
    normal = normal_non_unit / norm(normal_non_unit)

    d = dot(normal, r_test - r_plus[1,:])
    rho = r_test - d*normal

    l_plus = Array{Float64, 1}(undef, 3) # distance from point at P0 to rho_plus
    l_minus = Array{Float64, 1}(undef, 3) # distance from point at P0 to rho_minus
    P0 = Array{Float64, 1}(undef, 3) # distance from point at rho to orthogonal point on edge or edge extension
    R0 = Array{Float64, 1}(undef, 3) # distance between r_test and closest point on edge or edge extension
    R_plus = Array{Float64, 1}(undef, 3) # distance between r_test and r_plus
    R_minus = Array{Float64, 1}(undef, 3) # distance between r_test and r_minus
    rho_plus = Array{Float64, 2}(undef, 3, 3) # projections of r_plus on triangle plane
    rho_minus = Array{Float64, 2}(undef, 3, 3) # projections of r_minus on triangle plane
    I_hat = Array{Float64, 2}(undef, 3, 3) # unit vectors parallel to edges
    u_hat = Array{Float64, 2}(undef, 3, 3) # unit vector orthogonal to edge in triangle plane
    P0_hat = Array{Float64, 2}(undef, 3, 3) # unit vectors of P0
    for i in 1:3 # i represents triangle edge index
        rho_plus[i,:] = r_plus[i,:] - normal*dot(r_plus[i,:],normal)
        rho_minus[i,:] = r_minus[i,:] - normal*dot(r_minus[i,:],normal)
        I_hat[i,:] = (rho_plus[i,:] - rho_minus[i,:]) / norm(rho_plus[i,:] - rho_minus[i,:])
        u_hat[i,:] = cross(I_hat[i,:], normal)
        l_plus[i] = dot(rho_plus[i,:]-rho, I_hat[i,:])
        l_minus[i] = dot(rho_minus[i,:]-rho, I_hat[i,:])
        P0[i] = abs(dot(rho_plus[i,:] - rho, u_hat[i,:]))
        P0_hat[i,:] = (rho_plus[i,:]-rho-l_plus[i]*I_hat[i,:]) / P0[i]
        R0[i] = sqrt(P0[i]^2 + d^2)
        R_plus[i] = sqrt(norm(rho_plus[i,:]-rho)^2 + d^2)
        R_minus[i] = sqrt(norm(rho_minus[i,:]-rho)^2 + d^2)
    end
    (d, P0_hat, u_hat, P0, R0, R_plus, R_minus, l_plus, l_minus)
end
