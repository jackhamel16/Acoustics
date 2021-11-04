# dependencies: everything
include("src/includes.jl")

# This is the primary execution file for the acoustic scattering code by parsing an
# input file containing all of the desired run settings.  Examples of such files
# exist in the examples directory.  This is not a rigorous routine so if there are
# errors in the formatting of the input files, they will likely just cause errors in
# the code at runtime

if length(ARGS) == 1
    inputs_filename = ARGS[1]
    no_tag = ""

    println("Julia utilizing ", Threads.nthreads(), " threads")

    println("Parsing input parameters in: ", inputs_filename,"...")
    inputs = parseInputParams(inputs_filename)
    src_quadrature_rule = rule_lookup_dict[inputs.src_quadrature_string]
    test_quadrature_rule = rule_lookup_dict[inputs.test_quadrature_string]
    @unpack mesh_filename,
            equation,
            distance_to_edge_tol,
            near_singular_tol,
            excitation_params,
            ACA_params,
            WS_params = inputs

    println("Parsing mesh in: ", mesh_filename, "...")
    pulse_mesh = buildPulseMesh(mesh_filename, src_quadrature_rule, test_quadrature_rule)
    println("Number of elements = ",pulse_mesh.num_elements)
    # Excitation building, skipped if running WS mode
    if excitation_params.type == "planewave"
        println("Excitation: Plavewave")
        excitationFunc(x_test,y_test,z_test) = planeWave(excitation_params.amplitude,
                                                         excitation_params.wavevector,
                                                         [x_test,y_test,z_test])
        excitationFuncNormalDeriv(x_test,y_test,z_test,normal) = planeWaveNormalDerivative(excitation_params.amplitude,
                                                                                           excitation_params.wavevector,
                                                                                           [x_test,y_test,z_test],
                                                                                           normal)

    elseif excitation_params.type == "sphericalwave"
        println("Excitation: Sphericalwave")
        excitationFunc(x_test,y_test,z_test) = sphericalWave(excitation_params.amplitude,
                                                             excitation_params.wavenumber,
                                                             [x_test,y_test,z_test],
                                                             excitation_params.l,
                                                             excitation_params.m)
        excitationFuncNormalDeriv(x_test,y_test,z_test,normal) = sphericalWaveNormalDerivative(excitation_params.amplitude,
                                                                                               excitation_params.wavenumber,
                                                                                               [x_test,y_test,z_test],
                                                                                               excitation_params.l,
                                                                                               excitation_params.m,
                                                                                               normal)
    end

    run_time = 0.0
    # Determine which solver to run
    if equation == "sound soft IE"
        println("Equation: Sound Soft IE")
        if ACA_params.use_ACA == true
            println("Running with ACA...")
            run_time = @elapsed sources, octree, metrics = solveSoftIEACA(pulse_mesh,
                                                           ACA_params.num_levels,
                                                           excitationFunc,
                                                           excitation_params.wavenumber,
                                                           distance_to_edge_tol,
                                                           near_singular_tol,
                                                           ACA_params.compression_distance,
                                                           ACA_params.approximation_tol)
            printACAMetrics(metrics)
        else
            run_time = @elapsed sources = solveSoftIE(pulse_mesh,
                                  excitationFunc,
                                  excitation_params.wavenumber,
                                  distance_to_edge_tol,
                                  near_singular_tol)
        end
        exportSourcesBundled(mesh_filename, no_tag, sources)

    elseif equation == "sound soft normal derivative IE"
        println("Equation: Sound Soft Normal Derivative IE")
        if ACA_params.use_ACA == true
            println("ACA not implemented for this equation yet")
        else
            run_time = @elapsed sources = solveSoftIENormalDeriv(pulse_mesh,
                                             excitationFuncNormalDeriv,
                                             excitation_params.wavenumber)
        end
        exportSourcesBundled(mesh_filename, no_tag, sources)

    elseif equation == "sound soft CFIE"
        println("Equation: Sound Soft CFIE")
        if ACA_params.use_ACA == true
            run_time = @elapsed sources, octree, metrics = solveSoftCFIEACA(pulse_mesh,
                                                  ACA_params.num_levels,
                                                  excitationFunc,
                                                  excitationFuncNormalDeriv,
                                                  excitation_params.wavenumber,
                                                  inputs.CFIE_weight,
                                                  distance_to_edge_tol,
                                                  near_singular_tol,
                                                  ACA_params.compression_distance,
                                                  ACA_params.approximation_tol)
        printACAMetrics(metrics)
        else
            run_time = @elapsed sources = solveSoftCFIE(pulse_mesh,
                                    excitationFunc,
                                    excitationFuncNormalDeriv,
                                    excitation_params.wavenumber,
                                    distance_to_edge_tol,
                                    near_singular_tol,
                                    inputs.CFIE_weight)
        end
        exportSourcesBundled(mesh_filename, no_tag, sources)

    elseif equation == "WS mode"
        println("Equation: WS Mode") # this mode only runs sound soft IE right now
        if inputs.src_quadrature_string != inputs.test_quadrature_string
            println("When running at WS modes, test and src quadrature rules must be identical. Change settings and rerun.")
        else
            if ACA_params.use_ACA == true
                println("Running with ACA...")
                run_time = @elapsed sources, octree, metrics = solveWSModeSoftACA(WS_params,
                                                                              pulse_mesh,
                                                                              distance_to_edge_tol,
                                                                              near_singular_tol,
                                                                              ACA_params.num_levels,
                                                                              ACA_params.compression_distance,
                                                                              ACA_params.approximation_tol)
                printACAMetrics(metrics)
                # for local_mode_idx = 1:length(WS_params.mode_idxs)
                #     mode_idx = WS_params.mode_idxs[local_mode_idx]
                #     mode_tag = string("_mode", mode_idx)
                #     exportSourcesBundled(mesh_filename, mode_tag, sources[local_mode_idx])
                # end
            else
                run_time = @elapsed sources = solveWSModeSoft(WS_params, pulse_mesh,
                                      distance_to_edge_tol, near_singular_tol)
                # for local_mode_idx = 1:length(WS_params.mode_idxs)
                #     mode_idx = WS_params.mode_idxs[local_mode_idx]
                #     mode_tag = string("_mode", mode_idx)
                #     exportSourcesBundled(mesh_filename, mode_tag, sources[local_mode_idx])
                # end
            end
        end
    elseif equation == "WS mode CFIE"
        println("Equation: WS Mode CFIE") # this mode only runs sound soft IE right now
        if inputs.src_quadrature_string != inputs.test_quadrature_string
            println("When running at a WS mode, test and src quadrature rules must be identical. Change settings and rerun.")
        else
            if ACA_params.use_ACA == true
                println("Running with ACA...")
                run_time = @elapsed sources, octree, metrics = solveWSModeSoftCFIEACA(WS_params,
                                                                              pulse_mesh,
                                                                              distance_to_edge_tol,
                                                                              near_singular_tol,
                                                                              inputs.CFIE_weight,
                                                                              ACA_params.num_levels,
                                                                              ACA_params.compression_distance,
                                                                              ACA_params.approximation_tol)
                printACAMetrics(metrics)
                # for local_mode_idx = 1:length(WS_params.mode_idxs)
                #     mode_idx = WS_params.mode_idxs[local_mode_idx]
                #     mode_tag = string("_mode", mode_idx)
                #     exportSourcesBundled(mesh_filename, mode_tag, sources[local_mode_idx])
                # end
            else
                println("Not Implemented")# run_time = @elapsed sources = solveWSMode(WS_params.max_l, WS_params.mode_idxs,
                #                       WS_params.wavenumber, pulse_mesh,
                #                       distance_to_edge_tol, near_singular_tol)
                # for local_mode_idx = 1:length(WS_params.mode_idxs)
                #     mode_idx = WS_params.mode_idxs[local_mode_idx]
                #     mode_tag = string("_mode", mode_idx)
                #     exportSourcesBundled(mesh_filename, mode_tag, sources[local_mode_idx])
                # end
            end
        end
    end
    println("Total Runtime = ", run_time, " seconds")

elseif length(ARGS) > 1
    println("Please provide only one input filename as argument")

else
    println("No input filename provided")
end
