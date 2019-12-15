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

@eval Documenter.Writers.HTMLWriter function render_navbar(ctx, navnode, edit_page_link::Bool)
    @tags div header nav ul li a span img

    # The breadcrumb (navigation links on top)
    navpath = @show Documents.navpath(navnode)
    header_links = map(navpath) do nn
        title = mdconvert(pagetitle(ctx, nn); droplinks=true)
        nn.page === nothing ? li(a[".is-disabled"](title)) : li(a[:href => navhref(ctx, nn, navnode)](title))
    end
    header_links[end] = header_links[end][".is-active"]
    breadcrumb = nav[".breadcrumb"]() ### modified to not show breadcrumbs

    # The "Edit on GitHub" links and the hamburger to open the sidebar (on mobile) float right
    navbar_right = div[".docs-right"]
    
    ### custom code to add MyBinder link
    if edit_page_link
        pageurl = get(getpage(ctx, navnode).globals.meta, :EditURL, getpage(ctx, navnode).source)
        nbpath = foldl(replace,["src-staging"=>"src",".md"=>".ipynb"], init=pageurl)
        if @show isfile(nbpath)
            url = "https://mybinder.org/v2/gh/marius311/CMBLensing.jl/master?urlpath=lab/tree/$(basename(nbpath))"
            push!(navbar_right.nodes, a[".docs-right", :href => url](img[:src => "https://mybinder.org/badge_logo.svg"]()))
        end
    end
    ###
    
    # Set the logo and name for the "Edit on.." button.
    if edit_page_link && (ctx.settings.edit_link !== nothing) && !ctx.settings.disable_git
        host_type = Utilities.repo_host_from_url(ctx.doc.user.repo)
        if host_type == Utilities.RepoGitlab
            host = "GitLab"
            logo = "\uf296"
        elseif host_type == Utilities.RepoGithub
            host = "GitHub"
            logo = "\uf09b"
        elseif host_type == Utilities.RepoBitbucket
            host = "BitBucket"
            logo = "\uf171"
        else
            host = ""
            logo = "\uf15c"
        end
        hoststring = isempty(host) ? " source" : " on $(host)"

        pageurl = get(getpage(ctx, navnode).globals.meta, :EditURL, getpage(ctx, navnode).source)
        
        
        edit_branch = isa(ctx.settings.edit_link, String) ? ctx.settings.edit_link : nothing
        url = if Utilities.isabsurl(pageurl)
            pageurl
        else
            if !(pageurl == getpage(ctx, navnode).source)
                # need to set users path relative the page itself
                pageurl = joinpath(first(splitdir(getpage(ctx, navnode).source)), pageurl)
            end
            Utilities.url(ctx.doc.user.repo, pageurl, commit=edit_branch)
        end
        if url !== nothing
            edit_verb = (edit_branch === nothing) ? "View" : "Edit"
            title = "$(edit_verb)$hoststring"
            push!(navbar_right.nodes,
                a[".docs-edit-link", :href => url, :title => title](
                    span[".docs-icon.fab"](logo),
                    span[".docs-label.is-hidden-touch"](title)
                )
            )
        end
    end

    # Settings cog
    push!(navbar_right.nodes, a[
        "#documenter-settings-button.docs-settings-button.fas.fa-cog",
        :href => "#", :title => "Settings",
    ])

    # Hamburger on mobile
    push!(navbar_right.nodes, a[
        "#documenter-sidebar-button.docs-sidebar-button.fa.fa-bars.is-hidden-desktop",
        :href => "#"
    ])

    # Construct the main <header> node that should be the first element in div.docs-main
    header[".docs-navbar"](breadcrumb, navbar_right)
end



makedocs(
    sitename="CMBLensing.jl", 
    source = "src-staging",
    build = "latest",
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
        "api.md"
    ],
)
