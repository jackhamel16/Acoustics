using StaticArrays
using BenchmarkTools
using LinearAlgebra

m = [1 2; 3 4]

vec = SVector(5.0, 4.0, 1.5)
arr = StaticArray{Tuple{3},Int64,1} # this is just a data type
arr2 = SArray{Tuple{2, 3}}([1 2 3; 4 5 6])
mat = SMatrix{2,2}(m)

m2 = @SMatrix [1 2; 3 4]

n = 10
mtest = randn(n, n)
vtest = randn(n)

mstest = SMatrix{n,n}(mtest)
vstest = SVector{n}(vtest)

for i in 1:n
    @views test = mstest[:,i]
end
