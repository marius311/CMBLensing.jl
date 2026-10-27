# Compare the two ways of applying the adjoint using a normwise scale. A
# relative tolerance on the resulting scalar is ill-conditioned when the dot
# product nearly cancels. The 64-epsilon margin is a practical allowance for
# this fixture's block products, Fourier transforms and dot products; it is not
# a strict worst-case bound.
function equirect_adjoint_values(f, Cg, fC, g, ::Type{T}) where T
    lhs = f' * Cg
    rhs = fC * g
    scale = norm(f) * norm(Cg) + norm(fC') * norm(g)
    lhs, rhs, 64 * eps(T) * scale
end

function test_equirect_adjoint(f, C, g, ::Type{T}, mapbasis) where T
    Cg = C * g
    lhs, rhs, atol = equirect_adjoint_values(f, Cg, f' * C, g, T)
    @test abs(lhs - rhs) <= atol

    # Explicitly exercise a nearly cancelling dot product as well as the
    # randomly drawn fields above.
    x = mapbasis(Cg)
    f_orth = f - (dot(f, x) / dot(x, x)) * x
    lhs, rhs, atol = equirect_adjoint_values(f_orth, Cg, f_orth' * C, g, T)
    @test abs(lhs) <= 64 * eps(T) * norm(f_orth) * norm(Cg)
    @test abs(lhs - rhs) <= atol

    # A 0.1% error in the adjoint path should not pass this tolerance.
    lhs, _, atol = equirect_adjoint_values(x, Cg, x' * C, g, T)
    wrong_rhs = (x' * (T(1.001) * C)) * g
    @test abs(lhs - wrong_rhs) > atol
end

function test_equirect_nonsymmetric_adjoint(proj, ::Type{T}) where T
    Ny, Nx = proj.Ny, proj.Nx
    blocks = zeros(T, Ny, Ny, Nx ÷ 2 + 1)
    blocks[1, 2, 2] = one(T)
    C = BlockDiagEquiRect{AzFourier}(blocks, proj)
    fvalues, gvalues = zeros(T, Ny, Nx), zeros(T, Ny, Nx)
    for j in 1:Nx
        fvalues[1, j] = gvalues[2, j] = cos(T(2π) * T(j - 1) / T(Nx))
    end
    f, g = EquiRectMap(fvalues, proj), EquiRectMap(gvalues, proj)
    lhs, rhs, atol = equirect_adjoint_values(f, C * g, f' * C, g, T)
    @test abs(lhs - rhs) <= atol
    # This deliberately omits the transpose in the adjoint path.
    @test abs(lhs - (C * f)' * g) > atol
end
