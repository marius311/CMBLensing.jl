



    



# ### misc
# Cℓ_to_Cov(f::FlatField{P,T,M}, args...; kwargs...) where {P,T,M} = adapt(M, Cℓ_to_Cov(P,T,spin(f),args...; kwargs...))

# function pixwin(f::FlatField) 
#     @unpack θpix,P,T,k = fieldinfo(f)
#     Diagonal(FlatFourier{P,T}((pixwin.(θpix,k) .* pixwin.(θpix,k'))[1:end÷2+1,:]))
# end

# global_rng_for(::Type{<:FlatField{<:Any,<:Any,M}}) where {M} = global_rng_for(M)

# """
#     fixed_white_noise(rng, F)

# Like white noise but the amplitudes are fixed to unity, only the phases are
# random. Currently only implemented when F is a Fourier basis. Note that unlike
# [`white_noise`](@ref), fixed white-noise generated in EB and QU Fourier bases
# are not statistically the same.
# """
# fixed_white_noise(rng, F::Type{<:FlatFieldFourier}) =
#      exp.(im .* angle.(basis(F)(white_noise(rng,F)))) .* fieldinfo(F).Nside
