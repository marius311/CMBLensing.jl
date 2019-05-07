@init @require Weave="44d3d7a6-8a23-5bf8-98c5-b353f8df5ec9" begin

    const weave_hide_str = "weave_hide_line=true"

    function weave_pyplot_preexecute_hook(chunk)
        if get(chunk.options, :pyplot, false)
            chunk.content = """
                import PyPlot; PyPlot.clf() #$weave_hide_str
                $(chunk.content)
                if !isempty(PyPlot.gcf().get_axes()); PyPlot.gcf(); end #$weave_hide_str
            """
        end
        return chunk
    end
    
    function weave_pyplot_postexecute_hook(chunk)
        if get(chunk.options, :pyplot, false)
            chunk.result = map(chunk.result) do r
                if occursin(weave_hide_str, r.code)
                    Weave.ChunkOutput("", r.stdout, r.displayed, r.rich_output, r.figures)
                else
                    r
                end
            end
        end
        return chunk
    end

    function setup_weave_pyplot()
        if !(weave_pyplot_preexecute_hook in Weave.preexecute_hooks)
            Weave.push_preexecute_hook(weave_pyplot_preexecute_hook)
        end
        if !(weave_pyplot_postexecute_hook in Weave.postexecute_hooks)
            Weave.push_postexecute_hook(weave_pyplot_postexecute_hook);
        end
    end
    
end
