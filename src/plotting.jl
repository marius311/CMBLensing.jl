# FlatS0
function plot(f::FlatS0{T,P}, ax; kwargs...) where {T,Θ,N,P<:Flat{Θ,N}}
    pyimport(:seaborn)[:heatmap](f[:Tx]; ax=ax, xticklabels=false, yticklabels=false, square=true, kwargs...)
    ax[:set_title]("T map ($(N)x$(N) @ $(Θ)')")
end 
function plot(fs::AbstractVecOrMat{<:FlatS0}; plotsize=4, kwargs...)
    fig,axs = subplots(size(fs,1), size(fs,2); figsize=plotsize.*(size(fs,2),size(fs,1)), squeeze=false)
    for i=eachindex(fs)
        plot(fs[i], axs[i]; kwargs...)
    end
end
plot(f::FlatS0; kwargs...) = plot([f]; kwargs...)

# FlatS2
function plot(f::FlatS2{T,P}, axs; kwargs...) where {T,Θ,N,P<:Flat{Θ,N}}
    for (ax,k) in zip(axs,["E","B"])
        ax = pyimport(:seaborn)[:heatmap](f[Symbol("$(k)x")]; ax=ax, xticklabels=false, yticklabels=false, square=true, kwargs...)
        ax[:set_title]("$k map ($(N)x$(N) @ $(Θ)')")
    end
end
function plot(fs::AbstractVector{<:FlatS2}; plotsize=4, kwargs...)
    fig,axs = subplots(length(fs),2; figsize=(plotsize.*(2,length(fs))), squeeze=false)
    for i=1:length(fs)
        plot(fs[i], axs[i,:]; kwargs...)
    end
end
plot(f::FlatS2; kwargs...) = plot([f]; kwargs...)

# FieldTuple{<:FlatS0,<:FlatS2} (i.e., TEB)
function plot(f::Field2Tuple{<:FlatS0{T,P},<:FlatS2{T,P}}, axs; kwargs...) where {T,Θ,N,P<:Flat{Θ,N}}
    for (ax,k) in zip(axs,["T","E","B"])
        m = k=="T" ? f.f1[:Tx] : f.f2[Symbol("$(k)x")]
        ax = pyimport(:seaborn)[:heatmap](m; ax=ax, xticklabels=false, yticklabels=false, square=true, kwargs...)
        ax[:set_title]("$k map ($(N)x$(N) @ $(Θ)')")
    end
end
function plot(fs::AbstractVector{<:Field2Tuple{<:FlatS0,<:FlatS2}}; plotsize=4, kwargs...)
    fig,axs = subplots(length(fs),3; figsize=(plotsize.*(3,length(fs))), squeeze=false)
    for i=1:length(fs)
        plot(fs[i], axs[i,:]; kwargs...)
    end
end
plot(f::Field2Tuple{<:FlatS0,<:FlatS2}; kwargs...) = plot([f]; kwargs...)

function plot(m::Matrix{<:Real}; kwargs...)
    m[isinf(m)]=NaN
    pyimport(:seaborn)[:heatmap](m; mask=isnan(m), xticklabels=false, yticklabels=false, square=true, kwargs...)
end
