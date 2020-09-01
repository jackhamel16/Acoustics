#module element
#export

function compute_centroid(vertices::Array{Float64,2})
    [(vertices[1,1]+vertices[2,1]+vertices[3,1])/3,
     (vertices[1,2]+vertices[2,2]+vertices[3,2])/3,
     (vertices[1,3]+vertices[2,3]+vertices[3,3])/3]
end


# vertices=Array{Float64,2}(undef,3,3)
# #vertices=[i*j for i=1:3, for j=1:3]
# for i in 1:3, j in 1:3
#     vertices[i,j] = i*j
# end
# compute_centroid(vertices)

#end # module
