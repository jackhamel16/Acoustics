using LinearAlgebra

include("quadrature.jl")
include("mesh.jl")
include("greens_functions.jl")
include("fill.jl")
include("octree.jl")

include("ACA.jl")

wavenumber = 1.0+0.0im
src_quadrature_rule = gauss7rule
test_quadrature_rule = gauss7rule
distance_to_edge_tol = 1e-12
near_singular_tol = 1.0
approximation_tol = 1e-3
mesh_filename = "examples/test/rectangle_plate_8elements_symmetric.msh"
mesh_filename = "examples/simple/disjoint_triangles.msh"
pulse_mesh =  buildPulseMesh(mesh_filename, src_quadrature_rule, test_quadrature_rule)
num_levels = 3
octree = createOctree(num_levels, pulse_mesh)
test_node = octree.nodes[6]
src_node = octree.nodes[9]

@unpack num_elements = pulse_mesh
num_ele_test_node = length(test_node.element_idxs)
num_ele_src_node = length(src_node.element_idxs)
U = Array{ComplexF64, 2}(undef, num_ele_test_node, 1)
V = Array{ComplexF64, 2}(undef, 1, num_ele_src_node)
I1 = 1
R_tilde = zeros(ComplexF64, num_ele_test_node, num_ele_src_node) # change to not store entire matrix?
#initialization
global_test_ele_idx = test_node.element_idxs[I1]
global_src_ele_idxs = src_node.element_idxs
R_tilde[I1,:] = computeZArray(pulse_mesh, wavenumber, distance_to_edge_tol,
                         near_singular_tol, global_test_ele_idx, global_src_ele_idxs)
R_tilde_J1, J1 = findmax(abs.(R_tilde[I1,:]))
V[1,:] = R_tilde[I1,:] ./ R_tilde_J1
global_src_ele_idx = src_node.element_idxs[J1]
global_test_ele_idxs = test_node.element_idxs
R_tilde[:,J1] = computeZArray(pulse_mesh, wavenumber, distance_to_edge_tol,
                         near_singular_tol, global_src_ele_idx, global_test_ele_idxs)
U[:,1] = R_tilde[:,J1]
# end initialization
k = 2
Z_tilde_k = U * V
term1 = norm(U[:,k-1])*norm(V[k-1,:])
while norm(U[:,k-1])*norm(V[k-1,:]) > approximation_tol * norm(U[:,1:k-1]*V[1:k-1,:]) && k <=100
    global U
    global V
    global k
    Ik = findmax(abs.(R_tilde[:,J1]))[2]
    Z_Ik_row = zeros(ComplexF64, num_ele_src_node)
    global_test_ele_idx = test_node.element_idxs[Ik]
    for src_ele_idx = 1:num_ele_src_node
        global_src_ele_idx = src_node.element_idxs[src_ele_idx]
        Z_Ik_row[src_ele_idx] = computeZEntrySoundSoft(pulse_mesh, wavenumber, distance_to_edge_tol, near_singular_tol, global_test_ele_idx, global_src_ele_idx)
    end
    sum_uv_term = zeros(ComplexF64, num_ele_src_node)
    for idx = 1:k-1
        sum_uv_term += U[Ik,idx] * V[idx,:]
    end
    R_tilde[Ik,:] = Z_Ik_row - sum_uv_term
    R_tilde_Jk, Jk = findmax(abs.(R_tilde[Ik,:]))
    V = cat(V, transpose(R_tilde[Ik,:] ./ R_tilde_Jk), dims=1)
    Z_Jk_col = zeros(ComplexF64, num_ele_test_node)
    global_src_ele_idx = src_node.element_idxs[Jk]
    for test_ele_idx = 1:num_ele_test_node
        global_test_ele_idx = test_node.element_idxs[test_ele_idx]
        Z_Jk_col[test_ele_idx] = computeZEntrySoundSoft(pulse_mesh, wavenumber, distance_to_edge_tol, near_singular_tol, global_test_ele_idx, global_src_ele_idx)
    end
    sum_uv_term = zeros(ComplexF64, num_ele_test_node)
    for idx = 1:k-1
        sum_uv_term += V[idx,Jk] * U[:,idx]
    end
    R_tilde[:,Jk] = Z_Jk_col - sum_uv_term
    U = cat(U, R_tilde[:,Jk], dims=2)
    k += 1
end # while
