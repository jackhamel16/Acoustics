include("src/includes.jl")

if length(ARGS) == 1
    inputs_filename = ARGS[1]
    println("Parsing input parameters in: ", inputs_filename,"...")
    inputs = parseInputParams(inputs_filename)
    src_quadrature_rule = rule_lookup_dict[inputs.src_quadrature_string]
    test_quadrature_rule = rule_lookup_dict[inputs.test_quadrature_string]
    println("Parsing mesh in: ", inputs.mesh_filename, "...")

    pulse_mesh = buildPulseMesh(inputs.mesh_filename, src_quadrature_rule, test_quadrature_rule)


elseif length(ARGS) > 1
    println("Please provide only one input filename as argument")
else
    println("No input filename provided")
end
