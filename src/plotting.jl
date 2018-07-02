using PyPlot
import PyPlot: plot

export plot

plotsize₀ = 4

pretty_name(s::Symbol) = pretty_name(string(s))
pretty_name(s::String) = s[1:1]*" "*(Dict('x'=>"map",'l'=>"fourier")[s[2]])

# generic plotting some components of a FlatField
function _plot(f::FlatField{T,P}, ax, k, title, vlim; units=:deg, ticklabels=true, kwargs...) where {T,Θ,N,P<:Flat{Θ,N}}
    x = Θ*N/Dict(:deg=>60,:arcmin=>1)[units]/2
    extent = [-x,x,-x,x]
    (title == nothing) && (title="$(pretty_name(k)) ($(N)x$(N) @ $(Θ)')")
    (vlim == nothing) && (vlim=:sym)
    _plot(f[k]; ax=ax, extent=extent, title=title, vlim=vlim, kwargs...)
    if ticklabels
        @pydef type MyFmt <: pyimport(:matplotlib)[:ticker][:ScalarFormatter]
            __call__(self,v,p=nothing) = py"super"(MyFmt,self)[:__call__](v,p)*Dict(:deg=>"°",:arcmin=>"′")[units]
        end
        ax[:xaxis][:set_major_formatter](MyFmt())
        ax[:yaxis][:set_major_formatter](MyFmt())
        ax[:set_ylabel]("Dec")
        ax[:set_xlabel]("RA")
        ax[:tick_params](labeltop=false, labelbottom=true)
    else
        ax[:tick_params](labeltop=false, labelleft=false)
    end
end

# plotting a map
function _plot(m::AbstractMatrix{<:Real}; ax=gca(), title=nothing, vlim=:sym, cmap="RdBu_r", cbar=true, kwargs...)
    
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
       
    m[isinf.(m)]=NaN
    
    cax = ax[:matshow](m; vmin=vmin, vmax=vmax, cmap=cmap, rasterized=true, kwargs...)
    cbar && gcf()[:colorbar](cax,ax=ax)
    title!=nothing && ax[:set_title](title, y=1)
    ax
end

# plotting fourier coefficients 
function _plot(m::AbstractMatrix{<:Complex}; kwargs...)
    _plot(log10.(abs.(ifftshift(unfold(m)))); vlim=(nothing,nothing), cmap=nothing, kwargs...)
end



doc"""
    plot(f::Field; kwargs...)
    plot(fs::VecOrMat{\<:Field}; kwarg...)
    
Plotting fields. 
"""
plot(f::Field; kwargs...) = plot([f]; kwargs...)
function plot(fs::AbstractVecOrMat{F}; plotsize=plotsize₀, which=default_which(F), title=nothing, vlim=nothing, kwargs...) where {F<:Field}
    fs = fs[:,:]
    (which isa Vector) || (which = [which])
    if size(fs,2)==1
        which = reshape(which,1,:)
        (m,n) = size(fs,1), length(which)
    elseif size(fs,1)==1
        (m,n) = length(which), size(fs,2)
    else
        length(which)==1 || throw(ArgumentError("If plotting a matrix of fields, `which` must be a single key."))
        (m,n) = size(fs)
    end
    fig,axs = subplots(m, n; figsize=plotsize.*[1.4*n,m], squeeze=false)
    _plot.(fs,axs,which,title,vlim; kwargs...)
    tight_layout(w_pad=-10)
    fig,axs,which
end
default_which(::Type{<:FlatS0})  = [:Tx]
default_which(::Type{<:FlatS2})  = [:Ex,:Bx]
default_which(::Type{<:FlatS02}) = [:Tx,:Ex,:Bx]
default_which(::Any) = throw(ArgumentError("Must specify `which` by hand for $S field."))


doc"""
    animate(fields::Vector{\<:Vector{\<:Field}}; interval=50, motionblur=true, kwargs...)

"""
animate(f::AbstractVecOrMat{<:Field}; kwargs...) = animate([f]; kwargs...)
animate(annonate::Function, args...; kwargs...) = animate(args...; annonate=annonate, kwargs...)
function animate(fields::AbstractVecOrMat{<:AbstractVecOrMat{<:Field}}; interval=50, motionblur=true, annonate=nothing, filename=nothing, kwargs...)
    fig, axs, which = plot(first.(fields); kwargs...)
    motionblur = (motionblur == true) ? [0.1, 0.5, 1, 0.5, 0.1] : (motionblur == false) ? [1] : motionblur
    
    if (annonate!=nothing); annonate(fig,axs,which); end
    
    ani = pyimport("matplotlib.animation")[:FuncAnimation](fig, 
        i->begin
            for (f,ax,k) in tuple.(fields,axs,which)
                if length(f)>1
                    img = ax[:images][1]
                    img[:set_data](sum(x*f[mod1(i-j+1,length(f))][k] for (j,x) in enumerate(motionblur)) / sum(motionblur))
                end
            end
            first.(getindex.(axs,:images))[:]
        end, 
        1:maximum(length.(fields)[:]),
        interval=interval, blit=true
        )
    close()
    if filename!=nothing
        ani[:save](filename)
    end
    HTML(ani[:to_html5_video]())
end
