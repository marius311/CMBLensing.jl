# Disposable diagnostic-branch harness: run each source revision in a fresh
# child process using the Pkg.test environment, including its MKL preference.
root = dirname(@__DIR__)
testenv = dirname(Base.active_project())
for (label, sha) in (
    ("baseline", "5a95787ad4fd8780ed1bc9af2a192901d49f0379"),
    ("pr117", "680d99cb8f84ceefa9b31a0cb057817a14fa4683"),
    ("integration_before_pr118", "8c51631"),
    ("integration_after_pr118", "bccd2c6"),
    ("integration_failed_head", "eb2d73d11c0586011e1e60da8aa64bef9c6cb5ec"),
)
    println("DIAGNOSTIC START ", label, " ", sha)
    flush(stdout)
    run(`git -C $root checkout $sha -- src`)
    cmd = `$(Base.julia_cmd()) --project=$testenv --startup-file=no $(joinpath(@__DIR__, "bilinear_mkl_diagnostic.jl"))`
    run(addenv(cmd, "CMB_AUDIT_SOURCE" => label))
    println("DIAGNOSTIC DONE ", label)
    flush(stdout)
end
