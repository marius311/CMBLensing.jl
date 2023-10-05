# ensure in right directory and environment
cd(dirname(@__FILE__))
using Pkg
pkg"activate ."

##

using CMBLensing
using Documenter

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

# for debug build. for prod built, md files will have been deleted
run(`jupytext --to notebook --update "src/*.md"`)
for file in readdir("src")
    if endswith(file, "ipynb")
        file |> convert_to_markdown |> convert_equations
    end
end
rm("src-staging/index.md",force=true)
symlink("../../README.md","src-staging/index.md")


# # highlight output cells (i.e. anything withouout a language specified) white
# # note: "output" language is added by us in documenter.tpl
@eval Documenter.Writers.HTMLWriter function domify(dctx::DCtx, node::Node, c::MarkdownAST.CodeBlock)
    ctx, navnode, settings = dctx.ctx, dctx.navnode, dctx.settings
    language = c.info
    # function mdconvert(c::Markdown.Code, parent::MDBlockContext; settings::Union{HTML,Nothing}=nothing, kwargs...)
    @tags pre code
    language = Documenter.codelang(language)
    if language == "documenter-ansi" || language == "output"
        return pre(domify_ansicoloredtext(c.code, "language-output hljs"))
    elseif settings !== nothing && settings.prerender &&
           !(isempty(language) || language == "nohighlight")
        r = hljs_prerender(c, settings)
        r !== nothing && return r
    end
    class = isempty(language) ? "nohighlight" : "language-$(language)"
    return pre(code[".$(class) .hljs"](c.code))
end


# adds the MyBinder button on each page that is a notebook
@eval Documenter.Writers.HTMLWriter function render_navbar(ctx, navnode, edit_page_link::Bool)
    @tags div header nav ul li a span img

    # Hamburger on mobile
    navbar_left = a[
        "#documenter-sidebar-button.docs-sidebar-button.docs-navbar-link.fa-solid.fa-bars.is-hidden-desktop",
        :href => "#",
    ]

    # The breadcrumb (navigation links on top)
    navpath = Documenter.navpath(navnode)
    header_links = map(navpath) do nn
        dctx = DCtx(ctx, nn, true)
        title = domify(dctx, pagetitle(dctx))
        nn.page === nothing ? li(a[".is-disabled"](title)) : li(a[:href => navhref(ctx, nn, navnode)](title))
    end
    header_links[end] = header_links[end][".is-active"]
    breadcrumb = nav[".breadcrumb"](
        ul[".is-hidden-mobile"](header_links),
        ul[".is-hidden-tablet"](header_links[end]) # when on mobile, we only show the page title, basically
    )

    # The "Edit on GitHub" links and the hamburger to open the sidebar (on mobile) float right
    navbar_right = div[".docs-right"]

        ### custom code to add MyBinder link
        if edit_page_link
            pageurl = get(getpage(ctx, navnode).globals.meta, :EditURL, getpage(ctx, navnode).source)
            nbpath = foldl(replace,["src-staging"=>"src",".md"=>".ipynb"], init=pageurl)
            if isfile(nbpath)
                url = "https://mybinder.org/v2/gh/marius311/CMBLensing.jl/gh-pages?urlpath=lab/tree/$(basename(nbpath))"
                push!(navbar_right.nodes, a[".docs-right", :href => url](img[:src => "https://mybinder.org/badge_logo.svg"]()))
            end
        end
        ###
    

    # Set up the link to the root of the remote Git repository
    #
    # By default, we try to determine it from the configured remote. If that fails, the link
    # is not displayed. The user can also pass `repolink` to HTML to either disable it
    # (repolink = nothing) or override the link URL (if set to a string). In the latter case,
    # we try to figure out what icon and string we should use based on the URL.
    if !isnothing(ctx.settings.repolink) && (ctx.settings.repolink isa String || ctx.doc.user.remote isa Remotes.Remote)
        url, (host, logo) = if ctx.settings.repolink isa String
            ctx.settings.repolink, host_logo(ctx.settings.repolink)
        else # ctx.doc.user.remote isa Remotes.Remote
            Remotes.repourl(ctx.doc.user.remote), host_logo(ctx.doc.user.remote)
        end
        # repourl() can sometimes return a nothing (Remotes.URL)
        if !isnothing(url)
            repo_title = "View the repository" * (isempty(host) ? "" : " on $host")
            push!(navbar_right.nodes,
                a[".docs-navbar-link", :href => url, :title => repo_title](
                    span[".docs-icon.fa-brands"](logo),
                    span[".docs-label.is-hidden-touch"](isempty(host) ? "Repository" : host)
                )
            )
        end
    end
    # Add an edit link, with just an icon, but only on pages where edit_page_link is true.
    # Some pages, like search, are special and do not have a source file to link to.
    edit_page_link && edit_link(ctx, navnode) do logo, title, url
        push!(navbar_right.nodes,
            a[".docs-navbar-link", :href => url, :title => title](
                span[".docs-icon.fa-solid"](logo)
            )
        )
    end

    # Settings cog
    push!(navbar_right.nodes, a[
        "#documenter-settings-button.docs-settings-button.docs-navbar-link.fa-solid.fa-gear",
        :href => "#", :title => "Settings",
    ])

    # Collapse/Expand All articles toggle
    push!(navbar_right.nodes, a[
        "#documenter-article-toggle-button.docs-article-toggle-button.fa-solid.fa-chevron-up",
        :href=>"javascript:;", :title=>"Collapse all docstrings",
    ])

    # Construct the main <header> node that should be the first element in div.docs-main
    header[".docs-navbar"](navbar_left, breadcrumb, navbar_right)
end



makedocs(
    sitename="CMBLensing.jl", 
    source = "src-staging",
    format = Documenter.HTML(
        assets = ["assets/cmblensing.css"],
        disable_git = true,
    ),
    pages = [
        "index.md",
        "01_lense_a_map.md",
        "02_posterior.md",
        "03_joint_MAP_example.md",
        "04_from_python.md",
        "05_field_basics.md",
        "06_gpu.md",
        "precompilation.md",
        "api.md"
    ],
    remotes = nothing
)

if haskey(ENV, "IMAGE_NAME")
    open("build/Dockerfile","w") do io
        write(io,"FROM $(ENV["IMAGE_NAME"])")
    end
end

deploydocs(
    repo = "github.com/marius311/CMBLensing.jl.git",
    push_preview = true,
    forcepush = true
)
