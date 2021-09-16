# Master include file for project
# Only direct dependencies listed

include("math.jl")
include("ACA/ACA.jl")
include("code_structures/input_params.jl")
include("code_structures/mesh.jl") # depends on gmsh.jl but included in src file
include("code_structures/octree.jl") #dependencies: mesh.jl
include("excitation.jl") # dependencies: math.jl
include("quadrature.jl") # dependencies: mesh.jl
include("fill.jl") # dependencies: mesh.jl quadrature.jl
include("greens_functions.jl") # dependencies: mesh.jl quadrature.jl
include("solve.jl") # dependencies: fill.jl greens_functions.jl mesh.jl
include("ACA/ACA_fill.jl") # dependencies: ACA.jl greens_functions.jl mesh.jl octree.jl quadrature.jl
include("ACA/fast_solve.jl") # dependencies: ACA_fill.jl fill.jl mesh.jl ACA.jl octree.jl math.jl
include("scattering_matrix.jl") # dependencies: solve.jl fast_solve.jl
include("wigner_smith.jl") # dependencies: scattering_matrix.jl fast_solve.jl
