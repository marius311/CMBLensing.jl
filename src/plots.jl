
using RecipesBase, PlotUtils

@recipe function plot(m::FlatS0)
    
    @unpack (θpix, Nx, Ny) = m
    arr = reverse(Float16.(Array(m[:Ix])), dims=1)
    clim = quantile(abs.(arr[@. !isnan(arr)][:]),0.999)
    
    seriestype   :=  :heatmap
    aspect_ratio :=  :equal
    seriescolor  --> cgrad(:RdBu, rev=true)
    size         --> (1200, 500)
    xlims        --> (1, Nx) # this is because the default settings add padding in the x direction
    ylims        --> (1, Ny) # there is probably a better way of turning that off
    # xticks := tick_locations(Nx, θpix)
    # yticks := tick_locations(Ny, θpix)
    # tick_direction := :out
    # axes
    axis         --> true
    framestyle   --> :box 
    grid         --> true
    # labeling
    # xguide       --> L"x"
    # yguide       --> L"y"
    # colorbar
    # colorbar_title --> L"\mu \mathrm{K~amin}"
    clims        --> (-clim, clim)
    # title := L"\mathrm{Map}~(x, y)~[%$Nx \times %$Ny~@~%$θpix ^\prime]"
    hover        --> false
    
    arr
    
end

@recipe function plot(Cℓ::Cℓs)
    Cℓ.ℓ, Cℓ.Cℓ
end