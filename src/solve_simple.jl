include("fill.jl")
include("greens_functions.jl")
include("mesh.jl")

function solveSimple(mesh_filename::String,
                     excitation::Function,
                     wavenumber::Complex{Float64},
                     distance_to_edge_tol::Float64)

    src_quadrature_rule = gauss1rule
    test_quadrature_rule = gauss1rule
    pulse_mesh = buildPulseMesh(mesh_filename)
    println("Filling RHS...")
    rhs = rhsFill(pulse_mesh.num_elements, pulse_mesh.elements, pulse_mesh.nodes, excitation, test_quadrature_rule)
    function testIntegrand(r_test, src_nodes, is_singular)
        if is_singular == true
            # area = norm(cross(src_nodes[2,:]-src_nodes[1,:], src_nodes[3,:]-src_nodes[2,:]))/2
            # radius = sqrt(area/pi)
            # # radius = sqrt(0.0490137703458181/pi)
            # integral = 1/(2*im*wavenumber) * (1 - exp(-1*im*wavenumber*radius))
            # # println(integral)
            integral = scalarGreensSingularIntegral(wavenumber, r_test, src_nodes, src_quadrature_rule,
                                         distance_to_edge_tol)
            # global numerical_parts
            # append!(numerical_parts, numerical_part)
        else
            greensIntegrand(x, y, z) = scalarGreens(norm([x, y, z]-r_test), wavenumber)
            ### for testing, effect single poitn integration by evaluating at centroid and scaling by area
            # x,y,z = barycentric2Cartesian(src_nodes, [1/3,1/3,1/3])
            # area = norm(cross(src_nodes[2,:]-src_nodes[1,:], src_nodes[3,:]-src_nodes[2,:]))/2
            # integral = area * greensIntegrand(x,y,z)
            ###
            integral = integrateTriangle(src_nodes, greensIntegrand, src_quadrature_rule[:,1:3], src_quadrature_rule[:,4])
        end
        integral
    end
    println("Filling Matrix...")
    z_matrix = matrixFill(pulse_mesh.num_elements, pulse_mesh.elements, pulse_mesh.nodes, testIntegrand, test_quadrature_rule)
    println("Inverting Matrix...")
    source_vec = z_matrix \ rhs
    source_vec, z_matrix
end
