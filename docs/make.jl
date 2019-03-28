using Documenter
using CMBLensing

##

# convert Jupyter notebooks to markdown and do some simple preprocessing before
# feeding into Documenter.jl

function convert_to_markdown(file)
    run(`jupyter nbconvert src/$file --to markdown --template documenter.tpl --output-dir src-staging`)
    return "src-staging/$(replace(file, "ipynb"=>"md"))"
end

function convert_equations(file)
    contents = read(file, String)
    contents = replace(contents, r"\$\$(.*)\$\$" => s"""```math
    \g<1>
    ```""")
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

##

# highlight output cells (i.e. anything whiteout a language specified) white

import Documenter.Writers.HTMLWriter: mdconvert
using Markdown
using Documenter.Writers.HTMLWriter: MDBlockContext, @tags
function mdconvert(c::Markdown.Code, parent::MDBlockContext; kwargs...)
    @tags pre code
    language = isempty(c.language) ? "none" : c.language
    pre[".language-$(language)"](code[".language-$(language)"](c.code))
end


##
makedocs(
    sitename="CMBLensing.jl", 
    source = "src-staging",
    assets = ["assets/cmblensing.css"],
    pages = [
        "index.md",
        "lense_a_map.md",
        "joint_MAP_example.md",
        "api.md"
    ],
    repo = "https://github.com/marius311/CMBLensing.jl/blob/{commit}{path}#{line}"
)
