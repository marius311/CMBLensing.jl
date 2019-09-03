using CMBLensing

##

using Documenter
# ensure in right directory
cd(dirname(@__FILE__))

# convert Jupyter notebooks to markdown and do some simple preprocessing before
# feeding into Documenter.jl

function convert_to_markdown(file)
    run(`jupyter nbconvert src/$file --to markdown --template documenter.tpl --output-dir src-staging`)
    return "src-staging/$(replace(file, "ipynb"=>"md"))"
end

function convert_equations(file)
    contents = read(file, String)
    contents = replace(contents, r"\$\$(.*?)\$\$"s => s"""```math
    \g<1>
    ```""")
    contents = replace(contents, r"\* \$(.*?)\$" => s"* ``\g<1>``") # starting a line with inline math screws up tex2jax for some reason
    write(file, contents)
    return file
end

rm("src-staging", force=true, recursive=true)
mkdir("src-staging")
for file in readdir("src")
    if endswith(file, "ipynb")
        file |> convert_to_markdown |> convert_equations
    elseif !startswith(file, ".")
        cp("src/$file", "src-staging/$file")
    end
end
symlink("../../README.md","src-staging/index.md")


# highlight output cells (i.e. anything withouout a language specified) white

@eval Documenter.Writers.HTMLWriter function mdconvert(c::Markdown.Code, parent::MDBlockContext; kwargs...)
    @tags pre code
    language = isempty(c.language) ? "none" : c.language
    pre[".language-$(language)"](code[".language-$(language)"](c.code))
end

# adds the MyBinder button on each page that is a notebook

@eval Documenter.Writers.HTMLWriter function render_article(ctx, navnode)
    @tags article header footer nav ul li hr span a img

    header_links = map(Documents.navpath(navnode)) do nn
        title = mdconvert(pagetitle(ctx, nn); droplinks=true)
        nn.page === nothing ? li(title) : li(a[:href => navhref(ctx, nn, navnode)](title))
    end

    topnav = nav(ul(header_links))

    pageurl = get(getpage(ctx, navnode).globals.meta, :EditURL, getpage(ctx, navnode).source)
    nbpath = foldl(replace,["src-staging"=>"src",".md"=>".ipynb"], init=pageurl)
    if isfile(nbpath)
        url = "https://mybinder.org/v2/gh/marius311/CMBLensing.jl/master?urlpath=lab/tree/$(basename(nbpath))"
        push!(topnav.nodes, a[".edit-page", :href => url](img[:src => "https://mybinder.org/badge_logo.svg"]()))
    end
    
    art_header = header(topnav, hr(), render_topbar(ctx, navnode))

    # build the footer with nav links
    art_footer = footer(hr())
    if navnode.prev !== nothing
        direction = span[".direction"]("Previous")
        title = span[".title"](mdconvert(pagetitle(ctx, navnode.prev); droplinks=true))
        link = a[".previous", :href => navhref(ctx, navnode.prev, navnode)](direction, title)
        push!(art_footer.nodes, link)
    end

    if navnode.next !== nothing
        direction = span[".direction"]("Next")
        title = span[".title"](mdconvert(pagetitle(ctx, navnode.next); droplinks=true))
        link = a[".next", :href => navhref(ctx, navnode.next, navnode)](direction, title)
        push!(art_footer.nodes, link)
    end

    pagenodes = domify(ctx, navnode)
    article["#docs"](art_header, pagenodes, art_footer)
end

makedocs(
    sitename="CMBLensing.jl", 
    source = "src-staging",
    build = "latest",
    assets = ["assets/cmblensing.css"],
    pages = [
        "index.md",
        "01_lense_a_map.md",
        "02_posterior.md",
        "03_joint_MAP_example.md",
        "04_from_python.md",
        "05_field_basics.md",
        "api.md"
    ],
    html_disable_git = true
)
