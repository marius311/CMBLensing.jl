# Documenter's version postprocessor runs after it fetches gh-pages, but before
# it commits and pushes. Use that point to guard this path's prior publication
# and the live source ref, then run the normal version postprocessor.
struct GuardedVersions{T}
    versions::T
    source_root::String
end

function Documenter.determine_deploy_subfolder(decision, guarded::GuardedVersions)
    return Documenter.determine_deploy_subfolder(decision, guarded.versions)
end

function github_doc_build()
    event = get(ENV, "GITHUB_EVENT_NAME", "")
    ref = get(ENV, "GITHUB_REF", "")
    sha = get(ENV, "GITHUB_SHA", "")
    number = get(ENV, "GITHUB_RUN_NUMBER", "")
    preview = match(r"\Arefs/pull/([1-9][0-9]*)/merge\z", ref)
    tag = match(r"\Arefs/tags/(v[0-9]+\.[0-9]+\.[0-9]+(?:-[0-9A-Za-z.-]+)?)\z", ref)
    folder = if event == "push" && ref == "refs/heads/master"
        "dev"
    elseif event == "pull_request" && preview !== nothing
        "previews/PR$(preview.captures[1])"
    elseif event in ("push", "workflow_dispatch") && tag !== nothing &&
            tryparse(VersionNumber, tag.captures[1]) !== nothing &&
            Documenter.version_tag_strip_build(tag.captures[1]) == tag.captures[1]
        String(tag.captures[1])
    else
        error("Cannot verify documentation source for event $(repr(event)) and ref $(repr(ref))")
    end
    occursin(r"\A[0-9a-f]{40}\z", sha) || error("Missing or invalid GITHUB_SHA")
    occursin(r"\A[1-9][0-9]{0,19}\z", number) || error("Missing or invalid GITHUB_RUN_NUMBER")
    return (; ref, sha, number = parse(BigInt, number), folder)
end

function parse_doc_provenance(contents)
    m = match(r"\Av1\n(refs/[^\n]+)\n([0-9a-f]{40})\n([1-9][0-9]{0,19})\n\z", contents)
    m === nothing && error("Existing documentation provenance is malformed")
    return (; ref = m.captures[1], sha = m.captures[2], number = parse(BigInt, m.captures[3]))
end

function verify_doc_source(root, build)
    # The checkout's .git is bind-mounted into the docs container and belongs
    # to the runner. Trust only this exact checkout for these source commands.
    root = realpath(root)
    source_git = `$(Documenter.git()) -c safe.directory=$root -C $root`
    realpath(readchomp(`$source_git rev-parse --show-toplevel`)) == root ||
        error("Documentation source root is not the expected checkout")
    source_head = readchomp(`$source_git rev-parse HEAD`)
    # A tag event's GITHUB_SHA may name the tag object rather than its commit.
    # Resolve it from the checkout, then record the commit in the provenance.
    commit_expr = "$(build.sha)^{commit}"
    source_commit = try
        readchomp(pipeline(`$source_git rev-parse --verify $commit_expr`; stderr = devnull))
    catch
        error("Could not resolve GITHUB_SHA to a documentation source commit")
    end
    source_head == source_commit || error("Documentation checkout does not match GITHUB_SHA")
    istag = startswith(build.ref, "refs/tags/")
    peeled_ref = "$(build.ref)^{}"
    remote_ref = try
        if istag
            readchomp(pipeline(`$source_git ls-remote --exit-code origin $(build.ref) $peeled_ref`; stderr = devnull))
        else
            readchomp(pipeline(`$source_git ls-remote --exit-code --refs origin $(build.ref)`; stderr = devnull))
        end
    catch
        error("Could not verify the current documentation source ref (check Git authentication and remote availability)")
    end
    if istag
        # ls-remote reports an annotated tag's object and a second, peeled
        # commit line; a lightweight tag has only the commit line.
        refs = Dict{String, String}()
        for line in split(remote_ref, '\n')
            parts = split(line, '\t'; limit = 2)
            length(parts) == 2 || error("Unexpected response while verifying documentation source ref")
            haskey(refs, parts[2]) && error("Duplicate documentation source ref in remote response")
            refs[parts[2]] = parts[1]
        end
        object = get(refs, build.ref, "")
        peeled = get(refs, "$(build.ref)^{}", object)
        length(refs) in (1, 2) && all(key -> key in (build.ref, peeled_ref), keys(refs)) &&
            object != "" && peeled == source_commit &&
            (build.sha == object || build.sha == peeled) ||
            error("Documentation source ref has changed since this build started")
    else
        remote_ref == "$(build.sha)\t$(build.ref)" || error("Documentation source ref has changed since this build started")
    end
    return source_commit
end

function Documenter.postprocess_before_push(guarded::GuardedVersions; subfolder, devurl, deploy_dir, dirname)
    build = github_doc_build()
    subfolder == build.folder || error("Unexpected documentation deployment path: $(repr(subfolder))")
    git = Documenter.git()
    marker = "$subfolder/.docbuild-provenance"
    exists(path) = success(pipeline(`$git cat-file -e HEAD:$path`; stdout = devnull, stderr = devnull))
    unmarked = false
    previous = nothing
    if exists(marker)
        previous = parse_doc_provenance(read(`$git show HEAD:$marker`, String))
        previous.ref == build.ref || error("Existing documentation provenance has a different source ref")
        previous.number > build.number && error("A newer workflow run has already published this documentation path")
    elseif exists(subfolder)
        unmarked = true
    end

    # Match both the checkout and the current remote ref. The latter is checked
    # after the gh-pages fetch: any intervening publish then invalidates the
    # force-with-lease push, and a retry checks both again.
    source_commit = verify_doc_source(guarded.source_root, build)
    previous !== nothing && previous.number == build.number && previous.sha != source_commit &&
        error("The same workflow run has conflicting source revisions")
    unmarked && @warn "Existing documentation path has no provenance; recording this verified source revision" subfolder
    write(joinpath(deploy_dir, ".docbuild-provenance"), "v1\n$(build.ref)\n$source_commit\n$(build.number)\n")
    return Documenter.postprocess_before_push(guarded.versions; subfolder, devurl, deploy_dir, dirname)
end

# Documenter creates a fresh temporary checkout and fetches gh-pages for each
# deploydocs call. Retry only a push rejected because that checkout went stale;
# retrying the whole deployment incorporates the other publisher's changes.
function stale_docs_push(error, output)
    error isa ProcessFailedException || return false
    length(error.procs) == 1 || return false
    process = only(error.procs)
    command = process.cmd.exec
    length(command) == 6 || return false
    basename(command[1]) == "git" || return false
    command[2:end] == ["push", "-q", "--force-with-lease", "upstream", "HEAD:gh-pages"] || return false
    return occursin(r"(?m)^\s*! \[rejected\]\s+HEAD -> gh-pages \(stale info\)\s*$", output)
end

function deploy_with_retry(deploy; max_attempts = 4, retry_delay = attempt -> sleep(attempt + rand()))
    max_attempts >= 1 || throw(ArgumentError("max_attempts must be positive"))
    for attempt in 1:max_attempts
        retry = false
        mktemp() do _, output
            try
                # Git's rejection text is locale-dependent. Capture and replay
                # stderr so normal deployment warnings and errors stay visible.
                redirect_stderr(output) do
                    withenv("LC_ALL" => "C") do
                        deploy()
                    end
                end
            catch error
                flush(output)
                seekstart(output)
                messages = read(output, String)
                print(stderr, messages)
                if attempt < max_attempts && stale_docs_push(error, messages)
                    retry = true
                else
                    rethrow()
                end
            end
            if !retry
                flush(output)
                seekstart(output)
                print(stderr, read(output, String))
            end
        end
        retry || return
        @warn "Documentation deployment was rejected because gh-pages changed; fetching it again before retrying" attempt max_attempts
        retry_delay(attempt)
    end
end
