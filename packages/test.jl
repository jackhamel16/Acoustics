include("gmsh.jl")

num_elements = 2
num_coord_dims_solution = 3
nodes_per_triangle_solution = 3
nodes_solution = convert(Array{Float64}, transpose(hcat([0,0,0],[0,1,0],[1,1,0],[1,0,0])))
elements_solution = convert(Array{UInt64}, transpose(hcat([2,1,4],[2,4,3])))
centroids_solution = convert(Array{Float64}, transpose(hcat([1/3,1/3,0],[2/3,2/3,0])))

num_coord_dims = 3
nodes_per_triangle = 3

test_mesh_filename = "../../examples/simple/rectangle_plate.msh"
gmsh.initialize()
gmsh.open(test_mesh_filename)
nodes = reshapeMeshArray(gmsh.model.mesh.getNodes()[2], num_coord_dims)
