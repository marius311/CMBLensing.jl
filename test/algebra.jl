push!(LOAD_PATH, pwd()*"/src")

P = Flat{1,64}
T = Float64
g = FFTgrid(T,P)
a = FlatS0Map{T,P}(rand(64,64))

# vec inner
[a,a]' * [a,a]
[∂x,∂x]' * [∂x,∂x]

#vec outer
[a,a] * [a,a]'
[∂x,∂x] * [∂x,∂x]'

#vec inner op on fields
[∂x,∂x]' * [a,a]

#op broadcasting
[∂x,∂y]*a

#matrix*vector
[a a; a a] * [a, a]
[∂x ∂x; ∂x ∂x] * [a, a]
[∂x ∂x; ∂x ∂x] * [∂x, ∂x]

#vector matrix
[a a] * [a a; a a]
[∂x ∂x] * [∂x ∂x; ∂x ∂x]


#matrix*matrix
[a a; a a] * [a a; a a]
