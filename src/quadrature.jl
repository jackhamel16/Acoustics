# The gauss7 rule comes from:
#    https://www.math.unipd.it/~alvise/SETS_CUBATURE_TRIANGLE/dunavant/set_dunavant_barycentric.m
#    Note: ordering of the triangle nodes counter-clockwise
const gauss7points = [3.33333333333333314829616256247391e-01 3.33333333333333314829616256247391e-01 3.33333333333333314829616256247391e-01
                      4.70142064105100010440452251714305e-01 4.70142064105100010440452251714305e-01 5.97158717897999791190954965713900e-01
                      4.70142064105100010440452251714305e-01 5.97158717897999791190954965713900e-02 4.70142064105100010440452251714305e-01
                      5.97158717897999791190954965713900e-02 4.70142064105100010440452251714305e-01 4.70142064105100010440452251714305e-01
                      1.01286507323455995943639607048681e-01 1.01286507323455995943639607048681e-01 7.97426985353087980357145170273725e-01
                      1.01286507323455995943639607048681e-01 7.97426985353087980357145170273725e-01 1.01286507323455995943639607048681e-01
                      7.97426985353087980357145170273725e-01 1.01286507323455995943639607048681e-01 1.01286507323455995943639607048681e-01]
const gauss7weights = [2.25000000000000255351295663786004e-01
                       1.32394152788506136442236993389088e-01
                       1.32394152788506136442236993389088e-01
                       1.32394152788506136442236993389088e-01
                       1.25939180544827139529573400977824e-01
                       1.25939180544827139529573400977824e-01
                       1.25939180544827139529573400977824e-01]

function gaussQuadrature(scale_factor::Float64, func::Function, points::Array{Float64, 2}, weights::Array{Float64, 1})::Float64
    num_points = length(weights)
    quadrature_sum = 0
    for sum_idx in 1:num_points
        x, y, z = points[sum_idx,:]
        quadrature_sum += weights[sum_idx] * func(x, y, z)
    end
    scale_factor * quadrature_sum
end
