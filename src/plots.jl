
using RecipesBase, PlotUtils

@recipe function plot(m::FlatS0; which=:Ix)
    
    @unpack (θpix, Nx, Ny) = m
    if which == :Ix
        arr = reverse(Float16.(Array(m[:Ix])), dims=1)
    else
        arr = reverse(Float16.(log.(abs.(Array(ifftshift(m[:Il, full_plane=true]))))), dims=1)
    end
    clim = quantile(abs.(arr[@. !isnan(arr)][:]),0.999)
    
    seriestype   :=  :heatmap
    aspect_ratio :=  :equal
    seriescolor  --> cgrad(:RdBu, rev=true)
    # size         --> (1200, 500)
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
    (Cℓ.ℓ, Cℓ.Cℓ)
end

@recipe function plot(k::GetDistKDE{2}; clevels=[0.95,0.68], filled=false)

    if filled
        seriestype := :contourf
        levels := [PyArray(k.kde.getContourLevels(clevels)); prevfloat(Inf)]
        alpha := 0.5
    else
        seriestype := :contour
        levels := PyArray(k.kde.getContourLevels(clevels))
    end
    cbar := false
    
    return PyArray(k.kde.x), PyArray(k.kde.y), PyArray(k.kde.P)

end

@recipe function plot(k::GetDistKDE{1}; transform=identity)
    (PyArray(k.kde.x), transform.(PyArray(k.kde.P)))
end