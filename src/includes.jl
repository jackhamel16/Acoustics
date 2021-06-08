# Master include file for project
# Only direct dependencies listed

include("math.jl")
include("mesh.jl") # depends on gmsh.jl but included in src file
include("octree.jl") #dependencies: mesh.jl
include("excitation.jl") # dependencies: math.jl
include("quadrature.jl") # dependencies: mesh.jl
include("fill.jl") # dependencies: quadrature.jl
include("greens_functions.jl") # dependencies: mesh.jl quadrature.jl
include("solve.jl") # dependencies: fill.jl greens_functions.jl mesh.jl
include("scattering_matrix.jl") # dependencies: solve.jl and others within
include("wigner_smith.jl") # dependencies: scattering_matrix.jl

include("ACA.jl") # dependencies: mesh.jl quadrature.jl greens_functions.jl
