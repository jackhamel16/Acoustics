using BenchmarkTools

function fastest_innerprod(x::Array{T}, y::Array{T})::T where T
    ans::T = 0
    @fastmath @simd for i = 1:length(x)
        @inbounds ans += x[i] * y[i]
    end
    ans
end

function better_innerprod3(x::Array{T}, y::Array{T})::T where T
    @assert length(x) == length(y)
    fastest_innerprod(x, y)
end

function slow_sum(x, y)
    @assert length(x) == length(y)
    ans = 0
    for i = 1:length(x)
        ans += x[i] + y[i]
    end
    return ans
end

function fastest_sum(x::Array{T}, y::Array{T})::T where T
    ans::T = 0
    @fastmath @simd for i = 1:length(x)
        @inbounds ans += x[i] + y[i]
    end
    return ans
end

function better_sum(x::Array{T}, y::Array{T})::T where T
    @assert length(x) == length(y)
    fastest_sum(x, y)
end

function smoothing_max_slow(x)
    return -sum(x.*exp.(x))/sum(exp.(x))
end

function smoothing_max_fast(x::Array{T})::T where T
    num::T = 0
    den::T = 0
    @simd for i = 1:length(x)
        expx = exp(x[i])
        @inbounds num += x[i] * expx
        @inbounds den += expx
    end
    return num/den
end

f(x) = x*2 + x^2 + 4*x^5
slow(x) = x.*2 + x.^2 + 4 .*x.^5
fast(x) = @. x*2 + x^2 + 4*x^5


function swapsub!(X::Array{T}, Y::Array{T}, inds) where T
    @views @. X[inds] = xor(X[inds], Y[inds])
    @views @. Y[inds] = xor(X[inds], Y[inds])
    @views @. X[inds] = xor(X[inds], Y[inds])
end

X = [1; 2; 3]
Y = [4; 5; 6]

n = 10000
x = randn(n)
y = randn(n)

# @benchmark slow_innerprod(x, y)
