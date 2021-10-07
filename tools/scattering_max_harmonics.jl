radius = 1
# height = 2
num_elements = 3788
#
# area = 2*(2 * pi*radius^2 + 2*pi*radius*height) #doubled for slotted cyl
area = 4* pi *radius^2 # sphere
# r = 1
# area = pi * r^2 #plate
println("Area: ", area)

ele_area = area / num_elements
avg_edge_length = sqrt(4 * ele_area / sqrt(3))
println(avg_edge_length)
lambda = 5#10*avg_edge_length
println("min lambda = ",lambda)
max_l = ceil(2*radius*2*pi/lambda)
println("max degree, l = ", max_l)
num_harmonics = max_l^2 + 2*max_l + 1
println("num harmonics = ", num_harmonics)
