include("src/includes.jl")

if length(ARGS) == 1
    inputs_filename = ARGS[1]

    println("Parsing input parameters in: ", inputs_filename,"...")
    inputs = parseInputParams(inputs_filename)
    src_quadrature_rule = rule_lookup_dict[inputs.src_quadrature_string]
    test_quadrature_rule = rule_lookup_dict[inputs.test_quadrature_string]

    println("Parsing mesh in: ", inputs.mesh_filename, "...")
    pulse_mesh = buildPulseMesh(inputs.mesh_filename, src_quadrature_rule, test_quadrature_rule)

    # Excitation building, skipped if running WS mode
    if inputs.excitation_params.type == "planewave"
        println("Excitation: Plavewave")
        excitationFunc(x_test,y_test,z_test) = planeWave(inputs.excitation_params.amplitude,
                                                         inputs.excitation_params.wavevector,
                                                         [x_test,y_test,z_test])
    elseif inputs.excitation_params.type == "sphericalwave"
        println("Excitation: Sphericalwave")
        excitationFunc(x_test,y_test,z_test) = sphericalWave(inputs.excitation_params.amplitude,
                                                             inputs.excitation_params.wavenumber,
                                                             [x_test,y_test,z_test],
                                                             inputs.excitation_params.l,
                                                             inputs.excitation_params.m)
    end

    if inputs.equation == "sound soft IE"
        println("Equation: Sound Soft IE")
        if inputs.ACA_params.use_ACA == true
            println("Running with ACA...")
            sources, octree, metrics = solveSoundSoftIEACA(pulse_mesh,
                                                           inputs.ACA_params.num_levels,
                                                           excitationFunc,
                                                           inputs.excitation_params.wavenumber,
                                                           inputs.distance_to_edge_tol,
                                                           inputs.near_singular_tol,
                                                           inputs.ACA_params.compression_distance,
                                                           inputs.ACA_params.approximation_tol)
            printACAMetrics(metrics)
        else
            sources = solveSoftIE(pulse_mesh,
                                  excitationFunc,
                                  inputs.excitation_params.wavenumber,
                                  inputs.distance_to_edge_tol,
                                  inputs.near_singular_tol)
        end
        exportSourcesGmsh(inputs.mesh_filename, "sources_real", real.(sources))
        exportSourcesGmsh(inputs.mesh_filename, "sources_imag", imag.(sources))
        exportSourcesGmsh(inputs.mesh_filename, "sources_mag", abs.(sources))
    elseif inputs.equation == "WS mode"
        println("Equation: WS Mode")
        if inputs.src_quadrature_string != inputs.test_quadrature_string
            println("When running at a WS mode, test and src quadrature rules must be identical. Defaulting to src rule.")
            pulse_mesh.test_quadrature_rule = pulse_mesh.src_quadrature_rule
        end
        sources = solveWSMode(inputs.WS_params.max_l, inputs.WS_params.mode_idx,
                              inputs.WS_params.wavenumber, pulse_mesh,
                              inputs.distance_to_edge_tol, inputs.near_singular_tol)
        exportSourcesGmsh(inputs.mesh_filename, "sources_real", real.(sources))
        exportSourcesGmsh(inputs.mesh_filename, "sources_imag", imag.(sources))
        exportSourcesGmsh(inputs.mesh_filename, "sources_mag", abs.(sources))
    end

elseif length(ARGS) > 1
    println("Please provide only one input filename as argument")
else
    println("No input filename provided")
end
