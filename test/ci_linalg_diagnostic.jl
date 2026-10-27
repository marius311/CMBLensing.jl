module CILinalgDiagnostic
using LinearAlgebra, Random

function probe(label; vary_threads=false)
    previous = BLAS.get_num_threads()
    println("CI_DIAG stage=", label, " julia=", VERSION, " cpu=", Sys.CPU_NAME,
            " model=", Sys.cpu_info()[1].model, " BLAS=", BLAS.get_config(), " original_threads=", previous)
    try
        println("CI_DIAG openblas_core=", unsafe_string(ccall((:openblas_get_corename64_, Base.libblas_name), Cstring, ())))
    catch err
        println("CI_DIAG core query unavailable: ", sprint(showerror, err))
    end
    try
        for nt in (vary_threads ? unique((previous, 1, 2, 4)) : (previous,))
            BLAS.set_num_threads(nt)
            for T in (Float32, Float64), n in (10, 20, 32)
                X = randn(MersenneTwister(7), T, n, n)
                A = X' * X + T(n) * I
                identity = Matrix{T}(I, n, n)
                println("CI_DIAG nt=", nt, " T=", T, " n=", n,
                        " left=", norm(A \ A - identity) / norm(identity),
                        " right=", norm(A / A - identity) / norm(identity))
            end
            nodes = range(-0.5, 0.5; length=20)
            A = [x^i for x in nodes, i=0:9]
            b = sin.(nodes)
            coeff = A \ b
            reference = Float64.(big.(A) \ big.(b))
            println("CI_DIAG nt=", nt, " coeff_relerr=", norm(coeff-reference)/norm(reference),
                    " fit_maxerr=", maximum(abs.(A*coeff-b)))
        end
    finally
        BLAS.set_num_threads(previous)
    end
    flush(stdout)
end
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    CILinalgDiagnostic.probe("before package load"; vary_threads=true)
end
