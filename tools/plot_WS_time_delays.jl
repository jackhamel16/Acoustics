using Plots

function plotWSTimeDelays(time_delays_filename::String)
    # reads the time delays from the auto-generated output file titled
    #   "Wigner_Smith_time_delays.txt" that is generated when running a WS
    #   problem.  Assumes the time delays have a negligible imaginary component.
    println("\nFilename provided: ", time_delays_filename)
    file = open(time_delays_filename, "r")
    file_lines = split(read(file, String),"\n")
    num_times = length(file_lines) - 2
    time_lines = file_lines[2:num_times+1]
    times = Array{Float64,1}(undef, num_times)
    for time_idx = 1:num_times
        times[time_idx] = parse(Float64, split(time_lines[time_idx], " ")[2])
    end
    Plots.plot([i for i=1:num_times], times, label="", title="Wigner Smith Time Delays", xlabel="WS Mode Index", ylabel="Time (seconds)", size=(800,600))
    plot_filename = "WS_time_delays_plot"
    savefig(plot_filename)
    println("Plot of time delays saved as: ", plot_filename, ".png\n")
    return(times)
end

if length(ARGS) == 1
    times = plotWSTimeDelays(ARGS[1])
else
    println("\nImproper command line arguments provided.")
end
