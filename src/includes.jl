# Master include file for project
# Only direct dependencies listed

include("math.jl")
include("code_structures/mesh.jl") # depends on gmsh.jl but included in src file
include("code_structures/octree.jl") #dependencies: mesh.jl
include("excitation.jl") # dependencies: math.jl
include("quadrature.jl") # dependencies: mesh.jl
include("ACA/ACA.jl") # dependencies: mesh.jl quadrature.jl greens_functions.jl
include("fill.jl") # dependencies: quadrature.jl
include("greens_functions.jl") # dependencies: mesh.jl quadrature.jl
include("solve.jl") # dependencies: fill.jl greens_functions.jl mesh.jl
include("ACA/fast_solve.jl")
include("scattering_matrix.jl") # dependencies: solve.jl and others within
include("wigner_smith.jl") # dependencies: scattering_matrix.jl



# Left off updating file paths
