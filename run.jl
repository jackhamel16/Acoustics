# dependencies: everything
include("src/includes.jl")

# This is the primary execution file for the acoustic scattering code by parsing an
# input file containing all of the desired run settings.  Examples of such files
# exist in the examples directory.  This is not a rigorous routine so if there are
# errors in the formatting of the input files, they will likely just cause errors in
# the code at runtime

function exportSourcesBundled(mesh_filename::String, sources::AbstractArray{T, 1}) where T <: Number
    # bundled exportSourcesGmsh calls to clean up below
    exportSourcesGmsh(mesh_filename, "sources_real", real.(sources))
    exportSourcesGmsh(mesh_filename, "sources_imag", imag.(sources))
    exportSourcesGmsh(mesh_filename, "sources_mag", abs.(sources))
end

if length(ARGS) == 1
    inputs_filename = ARGS[1]

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

    # Determine which solver to run
    if equation == "sound soft IE"
        println("Equation: Sound Soft IE")
        if ACA_params.use_ACA == true
            println("Running with ACA...")
            sources, octree, metrics = solveSoundSoftIEACA(pulse_mesh,
                                                           ACA_params.num_levels,
                                                           excitationFunc,
                                                           excitation_params.wavenumber,
                                                           distance_to_edge_tol,
                                                           near_singular_tol,
                                                           ACA_params.compression_distance,
                                                           ACA_params.approximation_tol)
            printACAMetrics(metrics)
        else
            sources = solveSoftIE(pulse_mesh,
                                  excitationFunc,
                                  excitation_params.wavenumber,
                                  distance_to_edge_tol,
                                  near_singular_tol)
        end
        exportSourcesBundled(mesh_filename, sources)

    elseif equation == "sound soft normal derivative IE"
        println("Equation: Sound Soft Normal Derivative IE")
        if ACA_params.use_ACA == true
            println("ACA not implemented for this equation yet")
        else
            sources = solveSoftIENormalDeriv(pulse_mesh,
                                             excitationFuncNormalDeriv,
                                             excitation_params.wavenumber)
        end
        exportSourcesBundled(mesh_filename, sources)

    elseif equation == "sound soft CFIE"
        println("Equation: Sound Soft CFIE")
        if ACA_params.use_ACA == true
            println("ACA not implemented for this equation yet")
        else
            sources = solveSoftCFIE(pulse_mesh,
                                    excitationFunc,
                                    excitationFuncNormalDeriv,
                                    excitation_params.wavenumber,
                                    distance_to_edge_tol,
                                    near_singular_tol,
                                    inputs.CFIE_weight)
        end
        exportSourcesBundled(mesh_filename, sources)

    elseif equation == "WS mode"
        println("Equation: WS Mode") # this mode only runs sound soft IE right now
        if inputs.src_quadrature_string != inputs.test_quadrature_string
            println("When running at a WS mode, test and src quadrature rules must be identical. Change settings and rerun.")
        else
            sources = solveWSMode(WS_params.max_l, WS_params.mode_idx,
                                  WS_params.wavenumber, pulse_mesh,
                                  distance_to_edge_tol, near_singular_tol)
            exportSourcesBundled(mesh_filename, sources)
        end
    end

elseif length(ARGS) > 1
    println("Please provide only one input filename as argument")

else
    println("No input filename provided")
end
