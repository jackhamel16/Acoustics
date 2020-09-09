#module greens_functions
#export scalar_greens

function scalar_greens(R,k)
    exp(-im*k*abs(R))/(4*pi*abs(R))
end

#end # module
