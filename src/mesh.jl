include("../packages/gmsh.jl")
using ..gmsh

gmsh.initialize();

mesh_filename = "../examples/rectangle_plate.msh"
gmsh.open(mesh_filename)
nodes = gmsh.model.mesh.getNode()
