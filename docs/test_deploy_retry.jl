using Documenter
using Test

include("deploy_with_retry.jl")

# Use Documenter's actual deployment machinery, with a local bare repository
# instead of GitHub. A hook makes another publisher advance gh-pages after
# Documenter fetches it but before it pushes.
struct LocalDeploy <: Documenter.DeployConfig
    remote::String
    subfolder::String
    preview::Bool
end
Documenter.authentication_method(::LocalDeploy) = Documenter.HTTPS
Documenter.authenticated_repo_url(config::LocalDeploy) = config.remote
function Documenter.deploy_folder(config::LocalDeploy; branch, kwargs...)
    return Documenter.DeployDecision(;
        all_ok = true, branch, repo = config.remote,
        subfolder = config.subfolder, is_preview = config.preview,
    )
end

const git = Documenter.git()

function putfile(path, contents)
    mkpath(dirname(path))
    write(path, contents)
end

function fixture(dir)
    remote = joinpath(dir, "remote.git")
    writer = joinpath(dir, "other-publisher")
    source = joinpath(dir, "source")
    source_remote = joinpath(dir, "source.git")
    source_advance = joinpath(dir, "source-advance")
    hooks = joinpath(dir, "hooks")
    config = joinpath(dir, "gitconfig")
    state = joinpath(dir, "state")
    mkpath.((writer, source, hooks, state))
    run(`$git init -q --bare $source_remote`)
    run(`$git -C $source init -q --initial-branch=master`)
    run(`$git -C $source config user.name Source`)
    run(`$git -C $source config user.email source@example.invalid`)
    run(`$git -C $source config commit.gpgsign false`)
    run(`$git -C $source commit -q --allow-empty -m source`)
    sha = readchomp(`$git -C $source rev-parse HEAD`)
    run(`$git -C $source remote add origin $source_remote`)
    run(`$git -C $source push -q origin HEAD:refs/heads/master HEAD:refs/pull/102/merge`)
    run(`$git clone -q -b master $source_remote $source_advance`)
    run(`$git -C $source_advance config user.name Source`)
    run(`$git -C $source_advance config user.email source@example.invalid`)
    run(`$git -C $source_advance config commit.gpgsign false`)
    run(`$git init -q --bare $remote`)
    run(`$git -C $writer init -q --initial-branch=gh-pages`)
    run(`$git -C $writer config user.name Publisher`)
    run(`$git -C $writer config user.email publisher@example.invalid`)
    run(`$git -C $writer config commit.gpgsign false`)
    putfile(joinpath(writer, "dev", "index.html"), "old master")
    putfile(joinpath(writer, "previews", "PR11", "index.html"), "existing preview")
    putfile(joinpath(writer, "previews", "PR102", "obsolete.txt"), "old preview")
    putfile(joinpath(writer, "dev", ".docbuild-provenance"), "v1\nrefs/heads/master\n$sha\n1\n")
    putfile(joinpath(writer, "previews", "PR102", ".docbuild-provenance"), "v1\nrefs/pull/102/merge\n$sha\n1\n")
    putfile(joinpath(writer, "keep.txt"), "unrelated")
    run(`$git -C $writer add -A`)
    run(`$git -C $writer commit -q -m seed`)
    run(`$git -C $writer remote add upstream $remote`)
    run(`$git -C $writer push -q upstream HEAD:gh-pages`)
    putfile(joinpath(source, "build", "index.html"), "new docs")
    run(`$git config --file $config core.hooksPath $hooks`)
    putfile(joinpath(hooks, "post-commit"), raw"""#!/bin/sh
set -eu
count_file="$DOCS_TEST_STATE/commits"
n=0
if [ -f "$count_file" ]; then n=$(cat "$count_file"); fi
n=$((n + 1))
printf '%s' "$n" > "$count_file"
if [ "$DOCS_TEST_MODE" != always ] && { [ "$DOCS_TEST_MODE" != once ] && [ "$DOCS_TEST_MODE" != samepath ] || [ "$n" -ne 1 ]; }; then exit 0; fi
# Git exports the first publisher's repository to hooks; clear it before using
# the second publisher's checkout.
unset GIT_DIR GIT_WORK_TREE GIT_INDEX_FILE GIT_PREFIX
git -C "$DOCS_TEST_WRITER" fetch -q upstream gh-pages
git -C "$DOCS_TEST_WRITER" checkout -q -B gh-pages FETCH_HEAD
if [ "$DOCS_TEST_MODE" = samepath ]; then
    git --git-dir "$DOCS_TEST_SOURCE_REMOTE" update-ref "$DOCS_TEST_SOURCE_REF" "$DOCS_TEST_NEW_SHA"
    mkdir -p "$DOCS_TEST_WRITER/$DOCS_TEST_TARGET"
    printf 'newer docs' > "$DOCS_TEST_WRITER/$DOCS_TEST_TARGET/index.html"
    printf 'v1\n%s\n%s\n3\n' "$DOCS_TEST_SOURCE_REF" "$DOCS_TEST_NEW_SHA" > "$DOCS_TEST_WRITER/$DOCS_TEST_TARGET/.docbuild-provenance"
    rel="$DOCS_TEST_TARGET/index.html"
elif [ "$DOCS_TEST_MODE" = always ]; then
    rel="competing/step-$n.txt"
else
    rel="$DOCS_TEST_COMPETING_FILE"
fi
if [ "$DOCS_TEST_MODE" != samepath ]; then
    mkdir -p "$(dirname "$DOCS_TEST_WRITER/$rel")"
    printf 'other publisher %s' "$n" > "$DOCS_TEST_WRITER/$rel"
fi
git -C "$DOCS_TEST_WRITER" add -A
git -C "$DOCS_TEST_WRITER" -c core.hooksPath=/dev/null commit -q -m competitor
git -C "$DOCS_TEST_WRITER" -c core.hooksPath=/dev/null push -q upstream HEAD:gh-pages
""")
    putfile(joinpath(hooks, "pre-push"), raw"""#!/bin/sh
set -eu
count_file="$DOCS_TEST_STATE/pushes"
n=0
if [ -f "$count_file" ]; then n=$(cat "$count_file"); fi
printf '%s' "$((n + 1))" > "$count_file"
if [ "$DOCS_TEST_MODE" = auth ]; then
    printf '%s\n' 'fatal: Authentication failed' >&2
    exit 1
fi
""")
    chmod(joinpath(hooks, "post-commit"), 0o755)
    chmod(joinpath(hooks, "pre-push"), 0o755)
    return (; remote, writer, source, source_remote, source_advance, sha, hooks, config, state)
end

function with_fixture(f, fixture; mode, competing_file = "", preview = true, sha = fixture.sha, number = 2, next_sha = "", target = "")
    ref = preview ? "refs/pull/102/merge" : "refs/heads/master"
    withenv(
        "GIT_CONFIG_GLOBAL" => fixture.config, "GIT_CONFIG_NOSYSTEM" => "1",
        "GIT_TERMINAL_PROMPT" => "0", "DOCS_TEST_MODE" => mode,
        "DOCS_TEST_STATE" => fixture.state, "DOCS_TEST_WRITER" => fixture.writer,
        "DOCS_TEST_COMPETING_FILE" => competing_file,
        "GITHUB_EVENT_NAME" => preview ? "pull_request" : "push",
        "GITHUB_REF" => ref,
        "GITHUB_SHA" => sha, "GITHUB_RUN_NUMBER" => string(number),
        "DOCS_TEST_SOURCE_REMOTE" => fixture.source_remote, "DOCS_TEST_SOURCE_REF" => ref,
        "DOCS_TEST_NEW_SHA" => next_sha, "DOCS_TEST_TARGET" => target,
    ) do
        f()
    end
end

function local_deploy(fixture, folder; preview, max_attempts = 4)
    attempts = Ref(0)
    deploy = () -> begin
        attempts[] += 1
        Documenter.deploydocs(;
            root = fixture.source, target = "build", repo = "local",
            devbranch = "master", push_preview = true, forcepush = true,
            deploy_config = LocalDeploy(fixture.remote, folder, preview),
            versions = GuardedVersions(["stable" => "v^", "v#.#", "dev" => "dev"], fixture.source),
        )
    end
    return attempts, () -> deploy_with_retry(deploy; max_attempts, retry_delay = _ -> nothing)
end

function remote_file(fixture, path)
    return read(`$git --git-dir=$(fixture.remote) show gh-pages:$path`, String)
end
function remote_has(fixture, path)
    return success(pipeline(`$git --git-dir=$(fixture.remote) cat-file -e gh-pages:$path`; stderr = devnull))
end

function next_source_commit(fixture)
    run(`$git -C $(fixture.source_advance) commit -q --allow-empty -m next`)
    sha = readchomp(`$git -C $(fixture.source_advance) rev-parse HEAD`)
    run(`$git -C $(fixture.source_advance) push -q origin HEAD:refs/heads/candidate`)
    return sha
end

function remove_marker(fixture, folder)
    run(`$git -C $(fixture.writer) rm -q $folder/.docbuild-provenance`)
    run(`$git -C $(fixture.writer) commit -q -m unmarked`)
    run(`$git -C $(fixture.writer) push -q upstream HEAD:gh-pages`)
end

function synthetic_commit(fixture, tree, message, parents...)
    parent_args = reduce(vcat, (["-p", parent] for parent in parents); init = String[])
    return readchomp(`$git -C $(fixture.source) commit-tree -m $message $tree $parent_args`)
end

function capture_errors(f)
    return mktemp() do _, output
        redirect_stderr(f, output)
        flush(output)
        seekstart(output)
        read(output, String)
    end
end

@testset "documentation deployment retries" begin
    for (folder, preview, competitor) in [
        ("previews/PR102", true, "dev/concurrent.txt"),
        ("dev", false, "previews/PR77/index.html"),
    ]
        mktempdir() do dir
            setup = fixture(dir)
            attempts, deploy = local_deploy(setup, folder; preview)
            messages = capture_errors() do
                with_fixture(setup; mode = "once", competing_file = competitor, preview) do
                    deploy()
                end
            end
            @test attempts[] == 2
            @test occursin("(stale info)", messages)
            @test occursin("fetching it again before retrying", messages)
            @test remote_file(setup, "$folder/index.html") == "new docs"
            @test remote_file(setup, competitor) == "other publisher 1"
            @test remote_file(setup, "previews/PR11/index.html") == "existing preview"
            @test remote_file(setup, "keep.txt") == "unrelated"
            preview && @test !remote_has(setup, "previews/PR102/obsolete.txt")
        end
    end

    for (folder, preview) in [("previews/PR102", true), ("dev", false)]
        mktempdir() do dir
            setup = fixture(dir)
            newer_sha = next_source_commit(setup)
            attempts, deploy = local_deploy(setup, folder; preview)
            messages = capture_errors() do
                with_fixture(setup; mode = "samepath", preview, next_sha = newer_sha, target = folder) do
                    @test_throws ErrorException deploy()
                end
            end
            @test attempts[] == 2
            @test occursin("(stale info)", messages)
            @test occursin("newer workflow run", messages)
            @test remote_file(setup, "$folder/index.html") == "newer docs"
            ref = preview ? "refs/pull/102/merge" : "refs/heads/master"
            @test remote_file(setup, "$folder/.docbuild-provenance") == "v1\n$ref\n$newer_sha\n3\n"

            # A guarded rerun of the older job starts after the newer publish.
            attempts, deploy = local_deploy(setup, folder; preview)
            messages = capture_errors() do
                with_fixture(setup; mode = "none", preview) do
                    @test_throws ErrorException deploy()
                end
            end
            @test attempts[] == 1
            @test occursin("newer workflow run", messages)
            @test remote_file(setup, "$folder/index.html") == "newer docs"
        end
    end

    mktempdir() do dir
        setup = fixture(dir)
        remove_marker(setup, "previews/PR102")
        attempts, deploy = local_deploy(setup, "previews/PR102"; preview = true)
        messages = capture_errors() do
            with_fixture(setup; mode = "none") do
                deploy()
            end
        end
        @test attempts[] == 1
        @test occursin("no provenance", messages)
        @test remote_file(setup, "previews/PR102/.docbuild-provenance") == "v1\nrefs/pull/102/merge\n$(setup.sha)\n2\n"
    end

    mktempdir() do dir
        setup = fixture(dir)
        remove_marker(setup, "previews/PR102")
        newer_sha = next_source_commit(setup)
        run(`$git --git-dir=$(setup.source_remote) update-ref refs/pull/102/merge $newer_sha`)
        attempts, deploy = local_deploy(setup, "previews/PR102"; preview = true)
        messages = capture_errors() do
            with_fixture(setup; mode = "none") do
                @test_throws ErrorException deploy()
            end
        end
        @test attempts[] == 1
        @test occursin("source ref has changed", messages)
        @test !remote_has(setup, "previews/PR102/.docbuild-provenance")
        @test !remote_has(setup, "previews/PR102/index.html")
    end

    mktempdir() do dir
        setup = fixture(dir)
        putfile(joinpath(setup.writer, "previews", "PR102", ".docbuild-provenance"), "invalid\n")
        run(`$git -C $(setup.writer) add -A`)
        run(`$git -C $(setup.writer) commit -q -m malformed`)
        run(`$git -C $(setup.writer) push -q upstream HEAD:gh-pages`)
        attempts, deploy = local_deploy(setup, "previews/PR102"; preview = true)
        messages = capture_errors() do
            with_fixture(setup; mode = "none") do
                @test_throws ErrorException deploy()
            end
        end
        @test attempts[] == 1
        @test occursin("provenance is malformed", messages)
        @test remote_file(setup, "previews/PR102/.docbuild-provenance") == "invalid\n"
        @test !remote_has(setup, "previews/PR102/index.html")
    end

    mktempdir() do dir
        setup = fixture(dir)
        run(`$git --git-dir=$(setup.source_remote) update-ref -d refs/pull/102/merge`)
        attempts, deploy = local_deploy(setup, "previews/PR102"; preview = true)
        messages = capture_errors() do
            with_fixture(setup; mode = "none") do
                @test_throws ErrorException deploy()
            end
        end
        @test attempts[] == 1
        @test occursin("Could not verify the current documentation source ref", messages)
        @test !remote_has(setup, "previews/PR102/index.html")
    end

    # The CI container bind-mounts a checkout owned by the runner. Simulate
    # Git's ownership rejection and verify only the explicit source checkout
    # is trusted by the source guard.
    mktempdir() do dir
        setup = fixture(dir)
        with_fixture(setup; mode = "none") do
            withenv("GIT_TEST_ASSUME_DIFFERENT_OWNER" => "1") do
                git_with_ownership_check = Documenter.git()
                @test !success(pipeline(`$git_with_ownership_check -C $(setup.source) rev-parse HEAD`; stdout = devnull, stderr = devnull))
                @test isnothing(verify_doc_source(setup.source, github_doc_build()))
            end
        end
    end

    # A PR can retain the same head while its tested merge changes because
    # master moved. The exact merge ref must reject the old build in that case.
    mktempdir() do dir
        setup = fixture(dir)
        tree = readchomp(`$git -C $(setup.source) rev-parse "HEAD^{tree}"`)
        pr_head = synthetic_commit(setup, tree, "PR head", setup.sha)
        new_base = synthetic_commit(setup, tree, "new base", setup.sha)
        old_merge = synthetic_commit(setup, tree, "old merge", setup.sha, pr_head)
        new_merge = synthetic_commit(setup, tree, "new merge", new_base, pr_head)
        run(`$git -C $(setup.source) push -q origin $old_merge:refs/pull/102/merge $new_merge:refs/heads/candidate`)
        run(`$git -C $(setup.source) checkout -q --detach $old_merge`)
        run(`$git --git-dir=$(setup.source_remote) update-ref refs/pull/102/merge $new_merge`)
        @test split(readchomp(`$git -C $(setup.source) show -s --format=%P $old_merge`))[2] == pr_head
        @test split(readchomp(`$git -C $(setup.source) show -s --format=%P $new_merge`))[2] == pr_head
        attempts, deploy = local_deploy(setup, "previews/PR102"; preview = true)
        messages = capture_errors() do
            with_fixture(setup; mode = "none", sha = old_merge) do
                @test_throws ErrorException deploy()
            end
        end
        @test attempts[] == 1
        @test occursin("source ref has changed", messages)
        @test !remote_has(setup, "previews/PR102/index.html")
    end

    # Local documentation builds without GitHub variables still skip deployment.
    mktempdir() do dir
        setup = fixture(dir)
        withenv("GITHUB_EVENT_NAME" => nothing, "GITHUB_REF" => nothing,
                "GITHUB_SHA" => nothing, "GITHUB_RUN_NUMBER" => nothing) do
            @test isnothing(Documenter.deploydocs(;
                root = setup.source, target = "build", repo = "local", devbranch = "master",
                deploy_config = nothing,
                versions = GuardedVersions(["stable" => "v^", "v#.#", "dev" => "dev"], setup.source),
            ))
        end
    end

    mktempdir() do dir
        setup = fixture(dir)
        attempts, deploy = local_deploy(setup, "previews/PR102"; preview = true)
        messages = capture_errors() do
            with_fixture(setup; mode = "auth") do
                @test_throws ProcessFailedException deploy()
            end
        end
        @test attempts[] == 1
        @test occursin("Authentication failed", messages)
        @test !occursin("fetching it again before retrying", messages)
        @test read(joinpath(setup.state, "pushes"), String) == "1"
        @test !remote_has(setup, "previews/PR102/index.html")
    end

    mktempdir() do dir
        setup = fixture(dir)
        attempts, deploy = local_deploy(setup, "previews/PR102"; preview = true)
        messages = capture_errors() do
            with_fixture(setup; mode = "always") do
                @test_throws ProcessFailedException deploy()
            end
        end
        @test attempts[] == 4
        @test length(collect(eachmatch(r"\(stale info\)", messages))) == 4
        @test length(collect(eachmatch(r"fetching it again before retrying", messages))) == 3
        @test !remote_has(setup, "previews/PR102/index.html")
        for n in 1:4
            @test remote_file(setup, "competing/step-$n.txt") == "other publisher $n"
        end
        @test remote_file(setup, "previews/PR11/index.html") == "existing preview"
    end

    attempts = Ref(0)
    messages = capture_errors() do
        @test_throws ProcessFailedException deploy_with_retry(; retry_delay = _ -> nothing) do
            attempts[] += 1
            # Even a lookalike rejection must not retry a failure from another command.
            println(stderr, " ! [rejected] HEAD -> gh-pages (stale info)")
            run(`$git -C /nonexistent status`)
        end
    end
    @test attempts[] == 1
    @test occursin("cannot change to", messages)
    @test !occursin("fetching it again before retrying", messages)

    attempts[] = 0
    messages = capture_errors() do
        @test_throws ErrorException deploy_with_retry(; retry_delay = _ -> nothing) do
            attempts[] += 1
            println(stderr, " ! [rejected] HEAD -> gh-pages (stale info)")
            error("documentation build failed")
        end
    end
    @test attempts[] == 1
    @test !occursin("fetching it again before retrying", messages)
end
