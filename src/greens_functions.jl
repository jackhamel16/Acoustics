# dependencies: mesh.jl quadrature.jl

using LinearAlgebra

function scalarGreens(R::Float64, k::Number)
    exp(-im*k*abs(R))/(4*pi*abs(R))
end

function scalarGreensKDeriv(R, k)
    # derivative of 3D scalar Green's function w/ respect to k
    -im*exp(-im*k*R)/(4*pi)
end

function scalarGreensNonSingular(R::Float64, k::Number)
    (exp(-im*k*abs(R))-1)/(4*pi*abs(R))
end

function scalarGreensNormalDerivative(R_vec::AbstractArray{Float64, 1}, k::Number, nhat::AbstractArray{Float64, 1})
    R = norm(R_vec)
    grad_G = [-R_vec[1]*exp(-im*k*R)/(4*pi*R^3) - im*k*R_vec[1]*exp(-im*k*R)/(4*pi*R^2),
              -R_vec[2]*exp(-im*k*R)/(4*pi*R^3) - im*k*R_vec[2]*exp(-im*k*R)/(4*pi*R^2),
              -R_vec[3]*exp(-im*k*R)/(4*pi*R^3) - im*k*R_vec[3]*exp(-im*k*R)/(4*pi*R^2)]
    dot(nhat, grad_G)
end

@views function scalarGreensNormalDerivativeIntegration(pulse_mesh::PulseMesh,
                                 element_idx::Int64,
                                 wavenumber::Number,
                                 r_test::Array{Float64, 1},
                                 is_singular::Bool)
    @unpack nodes,
            elements,
            areas,
            normals,
            src_quadrature_rule,
            src_quadrature_points,
            src_quadrature_weights = pulse_mesh
    triangle_nodes = getTriangleNodes(element_idx, elements, nodes)
    triangle_area = areas[element_idx]
    if is_singular == true
        return(0.5) # principal value of integral for r=r'
    else
        scalar_greens_nd_integrand(x,y,z) = scalarGreensNormalDerivative([x,y,z]-r_test, wavenumber, normals[element_idx,:])
        return(gaussQuadrature(triangle_area, scalar_greens_nd_integrand,
                                 src_quadrature_points[element_idx], src_quadrature_weights))
    end
end

@views function scalarGreensIntegration(pulse_mesh::PulseMesh,
                                 element_idx::Int64,
                                 wavenumber::Number,
                                 r_test::Array{Float64, 1},
                                 distance_to_edge_tol::Float64,
                                 near_singular_tol::Float64,
                                 is_singular::Bool)
    # This function encapsulates all possible integration routines for the
    # scalar Green's function over a source triangle.
    # distance_to_edge_tol is distance at which the projection of r_test must be
    #    to an edge or extension to ignore that edge's contribution to the integral
    # near_singular_tol is the number of max edge lengths away r_test can be
    #    before doing purely numerical integration
    @unpack nodes,
            elements,
            areas,
            src_quadrature_rule,
            src_quadrature_points,
            src_quadrature_weights = pulse_mesh
    triangle_nodes = getTriangleNodes(element_idx, elements, nodes)
    triangle_area = areas[element_idx]
    max_edge_length = 0.0
    for edge_idx in 1:3
        edge_length = norm(triangle_nodes[edge_idx,:]-triangle_nodes[edge_idx%3+1,:])
        if edge_length > max_edge_length
            max_edge_length = edge_length
        end
    end
    centroid_src = barycentric2Cartesian(triangle_nodes, [1/3, 1/3, 1/3])
    if is_singular == true
        scalarGreensSingularIntegral(wavenumber, r_test, triangle_nodes,
                                     src_quadrature_rule[1:3,:], src_quadrature_weights,
                                     distance_to_edge_tol)
    elseif norm(r_test-centroid_src) > near_singular_tol*max_edge_length
        scalar_greens_integrand(x,y,z) = scalarGreens(norm([x,y,z]-r_test), wavenumber)
        gaussQuadrature(triangle_area, scalar_greens_integrand,
                        src_quadrature_points[element_idx], src_quadrature_weights)
    else
        scalarGreensNearSingularIntegral(wavenumber, r_test,
                                         triangle_nodes, triangle_area,
                                         src_quadrature_points[element_idx],
                                         src_quadrature_weights, distance_to_edge_tol)
    end
end

@views function computeScalarGreensSingularityIntegralParameters(r_test::Array{Float64, 1},
                                                          nodes::Array{Float64, 2})
    # Computes the input parameters for scalarGreensSingularityIntegral()
    # r_test is the observation point
    # nodes are the three triangle nodes
    r_minus = nodes # beginning points of edges
    r_plus = circshift(nodes, (-1,0)) # ending points of edges

    normal_non_unit = cross(r_plus[1,:]-r_minus[1,:], r_minus[3,:]-r_plus[3,:])
    normal = normal_non_unit / norm(normal_non_unit)

    d = dot(normal, r_test - r_plus[1,:])
    rho = r_test - normal * dot(r_test, normal)

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

function scalarGreensNearSingularIntegral(wavenumber::Number,
                                          r_test::Array{Float64, 1},
                                          nodes::Array{Float64, 2},
                                          triangle_area::Float64,
                                          quadrature_points::AbstractArray{Float64, 2},
                                          quadrature_weights::AbstractArray{Float64, 1},
                                          distance_to_edge_tol::Float64)
    # Used when r_test is close to source triangle, but not on it. Uses a
    # combination of analytical integration for the singular term and numerical
    # for the rest.
    non_singular_integrand(x, y, z) = scalarGreensNonSingular(norm([x,y,z]-r_test),
                                                            wavenumber)
    non_singular_integral = gaussQuadrature(triangle_area, non_singular_integrand,
                                            quadrature_points, quadrature_weights)
    singular_integral = scalarGreensSingularityIntegral(r_test, nodes,
                                                        distance_to_edge_tol)
    singular_integral + non_singular_integral
end

@views function scalarGreensSingularIntegral(wavenumber::Number,
                                             r_test::Array{Float64, 1},
                                             nodes::Array{Float64, 2},
                                             area_quadrature_points::AbstractArray{Float64, 2},
                                             quadrature_weights::AbstractArray{Float64, 1},
                                             distance_to_edge_tol::Float64)
    # Computes the integral of the scalar greens function for self-interactions
    # i.e. r_test is in the source triangle described by nodes
    total_integral = scalarGreensSingularityIntegral(r_test, nodes,
                                                     distance_to_edge_tol)
    sub_nodes = Array{Float64, 2}(undef, 4, 3)
    sub_nodes[1:3,:] = nodes
    sub_nodes[4,:] = r_test
    sub_elements = [4 2 3; 1 4 3; 1 2 4]
    quadrature_points = calculateQuadraturePoints(sub_nodes, sub_elements, area_quadrature_points)
    for triangle_idx in 1:3
        # integrate scalar greens function with singularity subtracted over sub triangles
        non_singular_integrand(x, y, z) = scalarGreensNonSingular(norm([x,y,z]-r_test), wavenumber)
        sub_element_area = calculateTriangleArea(sub_nodes[sub_elements[triangle_idx,:],:])
        total_integral += gaussQuadrature(sub_element_area,
                                          non_singular_integrand,
                                          quadrature_points[triangle_idx],
                                          quadrature_weights)
    end
    total_integral
end

@views function scalarGreensKDerivIntegration(pulse_mesh::PulseMesh,
                                 element_idx::Int64,
                                 wavenumber::Number,
                                 r_test::Array{Float64, 1},
                                 is_singular::Bool)
    # This function encapsulates all possible integration routines for the
    # scalar Green's function over a source triangle.
    # distance_to_edge_tol is distance at which the projection of r_test must be
    #    to an edge or extension to ignore that edge's contribution to the integral
    # near_singular_tol is the number of max edge lengths away r_test can be
    #    before doing purely numerical integration
    @unpack nodes,
            elements,
            areas,
            src_quadrature_rule,
            src_quadrature_points,
            src_quadrature_weights = pulse_mesh
    triangle_nodes = getTriangleNodes(element_idx, elements, nodes)
    triangle_area = areas[element_idx]
    if is_singular == true
        scalarGreensKDerivSingularIntegral(wavenumber, r_test, triangle_nodes,
                                           src_quadrature_rule[1:3,:], src_quadrature_weights)
    else
        greens_k_deriv_integrand(x,y,z) = scalarGreensKDeriv(norm([x,y,z]-r_test), wavenumber)
        gaussQuadrature(triangle_area, greens_k_deriv_integrand,
                          src_quadrature_points[element_idx], src_quadrature_weights)
    end
end

@views function scalarGreensKDerivSingularIntegral(wavenumber::Number,
                                                   r_test::Array{Float64, 1},
                                                   nodes::Array{Float64, 2},
                                                   area_quadrature_points::AbstractArray{Float64, 2},
                                                   quadrature_weights::AbstractArray{Float64, 1})
    # Computes the integral of the scalar greens function for self-interactions
    # i.e. r_test is in the source triangle described by nodes
    sub_nodes = Array{Float64, 2}(undef, 4, 3)
    sub_nodes[1:3,:] = nodes
    sub_nodes[4,:] = r_test
    sub_elements = [4 2 3; 1 4 3; 1 2 4]
    quadrature_points = calculateQuadraturePoints(sub_nodes, sub_elements, area_quadrature_points)
    total_integral = 0.0+im*0.0
    for triangle_idx in 1:3
        sub_element_nodes = getTriangleNodes(triangle_idx, sub_elements, sub_nodes)
        sub_element_area = calculateTriangleArea(sub_element_nodes)
        k_deriv_integrand(x, y, z) = scalarGreensKDeriv(norm([x,y,z]-r_test),
                                                             wavenumber)
        total_integral += gaussQuadrature(sub_element_area, k_deriv_integrand,
                                            quadrature_points[triangle_idx],
                                            quadrature_weights)
    end
    total_integral
end
