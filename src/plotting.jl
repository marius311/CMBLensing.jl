
using .PyPlot
using .PyPlot.PyCall
import .PyPlot: loglog, plot, semilogx, semilogy


### plotting Cℓs

for plot in (:plot, :loglog, :semilogx, :semilogy)
    @eval function ($plot)(ic::InterpolatedCℓs, args...; kwargs...)
		($plot)(ic.ℓ, ic.Cℓ, args...; kwargs...)
	end
	@eval function ($plot)(ic::InterpolatedCℓs{<:Measurement}, args...; kwargs...)
		errorbar(ic.ℓ, Measurements.value.(ic.Cℓ), Measurements.uncertainty.(ic.Cℓ), args...; marker=".", ls="", capsize=2, kwargs...)
		($plot) in [:loglog,:semilogx] && xscale("log")
		($plot) in [:loglog,:semilogy] && yscale("log")
	end
end


### plotting FlatFields

plotsize₀ = 4

pretty_name(s::Symbol) = pretty_name(Val.(Symbol.(split(string(s),"")))...)
pretty_name(::Val{s},::Val{:x}) where {s} = "$s Map"
pretty_name(::Val{s},::Val{:l}) where {s} = "$s Fourier"
pretty_name(::Val{:T},::Val{:x}) where {s} = "Map"
pretty_name(::Val{:T},::Val{:l}) where {s} = "Fourier"

# generic plotting some components of a FlatField
function _plot(f::FlatField{P}, ax, k, title, vlim; units=:deg, ticklabels=true, axeslabels=false, kwargs...) where {N,θ,P<:Flat{N,θ}}
    if string(k)[2] == 'x'
        x = θ*N/Dict(:deg=>60,:arcmin=>1)[units]/2
    elseif string(k)[2] == 'l'
        x = fieldinfo(f).nyq
    else
        throw(ArgumentError("Invalid `which`: $k"))
    end
    extent = [-x,x,-x,x]
    (title == nothing) && (title="$(pretty_name(k)) ($(N)x$(N) @ $(θ)')")
    (vlim == nothing) && (vlim=:sym)
    _plot(Array(f[k]); ax=ax, extent=extent, title=title, vlim=vlim, kwargs...)
    if ticklabels
        if string(k)[2] == 'x'
            @pydef mutable struct MyFmt <: pyimport(:matplotlib).ticker.ScalarFormatter
                __call__(self,v,p=nothing) = py"super"(MyFmt,self).__call__(v,p)*Dict(:deg=>"°",:arcmin=>"′")[units]
            end
            ax.xaxis.set_major_formatter(MyFmt())
            ax.yaxis.set_major_formatter(MyFmt())
            if axeslabels
                ax.set_xlabel("RA")
                ax.set_ylabel("Dec")
            end
        else
            ax.set_xlabel(raw"$\ell_x$")
            ax.set_ylabel(raw"$\ell_y$")
            ax.tick_params(axis="x", rotation=45)
        end
        ax.tick_params(labeltop=false, labelbottom=true)
    else
        ax.tick_params(labeltop=false, labelleft=false)
    end
end

# plotting a map
function _plot(m::AbstractMatrix{<:Real}; ax=gca(), title=nothing, vlim=:sym, cmap="RdBu_r", vscale=:linear, cbar=true, kwargs...)
    
    # some logic to automatically get upper/lower limits
    if vlim==:sym
        vmax = quantile(abs.(m[@. !isnan(m)][:]),0.999)
        vmin = -vmax
    elseif vlim==:asym
        vmin, vmax = (quantile(m[@. !isnan(m)][:],q) for q=(0.001,0.999))
    elseif isa(vlim,Tuple)
        vmin, vmax = vlim
    else
        vmax = vlim
        vmin = -vmax
    end
       
    m = Float64.(m)
    m[isinf.(m)] .= NaN
    
    cax = ax.matshow(clamp.(m,vmin,vmax); vmin=vmin, vmax=vmax, cmap=cmap, rasterized=true, kwargs...)
    cbar && gcf().colorbar(cax,ax=ax)
    title!=nothing && ax.set_title(title, y=1)
    ax
end

# plotting fourier coefficients 
function _plot(m::AbstractMatrix{<:Complex}; vscale=:log, kwargs...)
    dat = ifftshift(unfold(m))
    if vscale==:log
        dat .= log10.(abs.(dat))
    end
    _plot(real.(dat); vlim=(nothing,nothing), cmap=nothing, kwargs...)
end



@doc doc"""
    plot(f::Field; kwargs...)
    plot(fs::VecOrMat{\<:Field}; kwarg...)
    
Plotting fields. 
"""
plot(f::Field; kwargs...) = plot([f]; kwargs...)
function plot(fs::AbstractVecOrMat{F}; plotsize=plotsize₀, which=default_which(fs), title=nothing, vlim=nothing, return_all=false, kwargs...) where {F<:Field}
    (m,n) = size(tuple.(fs, which)[:,:])
    fig,axs = subplots(m, n; figsize=plotsize.*[1.4*n,m], squeeze=false)
    axs = getindex.(Ref(axs), 1:m, (1:n)') # see https://github.com/JuliaPy/PyCall.jl/pull/487#issuecomment-456998345
    _plot.(fs,axs,which,title,vlim; kwargs...)
    tight_layout(w_pad=-10)
    return_all ? (fig,axs,which) : isjuno ? fig : nothing
end
default_which(::AbstractVecOrMat{<:FlatS0}) = [:Ix]
default_which(::AbstractVecOrMat{<:FlatS2}) = [:Ex :Bx]
default_which(::AbstractVecOrMat{<:FlatS02}) = [:Ix :Ex :Bx]
function default_which(fs::AbstractVecOrMat{<:Field})
    try
        ensuresame((default_which([f]) for f in fs)...)
    catch x
        x isa AssertionError ? throw(ArgumentError("Must specify `which` argument by hand for this combination of fields to plot.")) : rethrow()
    end
end


@doc doc"""
    animate(fields::Vector{\<:Vector{\<:Field}}; interval=50, motionblur=false, kwargs...)

"""
animate(f::AbstractVecOrMat{<:Field}; kwargs...) = animate([f]; kwargs...)
animate(annonate::Function, args...; kwargs...) = animate(args...; annonate=annonate, kwargs...)
function animate(fields::AbstractVecOrMat{<:AbstractVecOrMat{<:Field}}; fps=25, motionblur=false, annonate=nothing, filename=nothing, kwargs...)
    fig, axs, which = plot(first.(fields); return_all=true, kwargs...)
    motionblur = (motionblur == true) ? [0.1, 0.5, 1, 0.5, 0.1] : (motionblur == false) ? [1] : motionblur
    
    if (annonate!=nothing); annonate(fig,axs,which); end
    
    ani = pyimport("matplotlib.animation").FuncAnimation(fig, 
        i->begin
            for (f,ax,k) in tuple.(fields,axs,which)
                if length(f)>1
                    img = ax.images[1]
                    img.set_data(sum(x*getindex(f[mod1(i-j+1,length(f))],k) for (j,x) in enumerate(motionblur)) / sum(motionblur))
                end
            end
            first.(getproperty.(axs,:images))[:]
        end, 
        1:maximum(length.(fields)[:]),
        interval=1000/fps, blit=true
        )
    close()
    if filename!=nothing
        ani.save(filename,writer="imagemagick",savefig_kwargs=Dict(:facecolor=>fig.get_facecolor()))
        if endswith(filename,".gif")
            run(`convert -layers Optimize $filename $filename`)
        end
    end
    HTML(ani.to_html5_video())
end
