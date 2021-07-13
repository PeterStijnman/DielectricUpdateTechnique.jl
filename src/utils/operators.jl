"""
compute the kronecker product of a and b.\n
used as a shorthand notation.\n
"""
⊗(a, b) = kron(a, b)

"""
AToE = _electric_vector_potential_to_electric_field_operator(number_of_cells, resolution, k_b)\n
This operators is used to compute the total electric field on the FDTD grid (Yee cell) given the vector potential.\n
k_b is the wavenumber in the background medium.\n
"""
function _electric_vector_potential_to_electric_field_operator(
    number_of_cells,
    resolution,
    k_b
)
    T = ComplexF32
    p = number_of_cells[1]
    q = number_of_cells[2]
    r = number_of_cells[3]
    Δx = Float32(resolution[1])
    Δy = Float32(resolution[2])
    Δz = Float32(resolution[3])

    kb = Float32(k_b)
    # needed -> Ex : (P+2) x (Q+1) : (R+1), wanted -> Ex : P x (Q+1) : (R+1)
    # needed -> Ey : (P+1) x (Q+2) : (R+1), wanted -> Ey : (P+1) x Q : (R+1)
    # needed -> Ez : (P+1) x (Q+1) : (R+2), wanted -> Ez : (P+1) x (Q+1) : R
    #operators for Ex
    Rp = spdiagm(p, p+2, 1=>ones(T, p))
    Rq = spdiagm(q+1, q+1, 0=>ones(T, q+1))
    Rr = spdiagm(r+1, r+1, 0=>ones(T, r+1))

    Ix   = (Rr ⊗ (Rq ⊗ Rp))
    ∂x   = 1/Δx .*spdiagm(p, p+1, 0 =>-one(T) .*ones(T, p), 1 =>ones(T, p))
    ∂y   = 1/Δy .*spdiagm(q+1, q+2, 0 =>-one(T) .*ones(T, q+1), 1 =>ones(T, q+1))
    ∂z   = 1/Δz .*spdiagm(r+1, r+2, 0 =>-one(T) .*ones(T, r+1), 1 =>ones(T, r+1))
    ∂x²  = 1/(Δx)^2 .*spdiagm(p, p+2, 0 => ones(T, p),1=> -2 .*ones(T, p), 2 => ones(T, p))

    xx = Ix'*((Rr ⊗ (Rq ⊗ ∂x²)) + kb^2 .*Ix)
    yx = Ix'*((Rr ⊗ (∂y ⊗ ∂x)))
    zx = Ix'*((∂z ⊗ (Rq ⊗ ∂x)))

    #operators for Ey
    Rp = spdiagm(p+1, p+1, 0=>ones(T, p+1))
    Rq = spdiagm(q, q+2, 1=>ones(T, q))
    Rr = spdiagm(r+1, r+1, 0=>ones(T, r+1))

    Iy   = (Rr ⊗ (Rq ⊗ Rp))
    ∂x   = 1/Δx .*spdiagm(p+1, p+2, 0 =>-one(T) .*ones(T, p+1), 1 =>ones(T, p+1))
    ∂y   = 1/Δy .*spdiagm(q, q+1, 0 =>-one(T) .*ones(T, q), 1 =>ones(T, q))
    ∂z   = 1/Δz .*spdiagm(r+1, r+2, 0 =>-one(T) .*ones(T, r+1), 1 =>ones(T, r+1))
    ∂y²  = 1/(Δy)^2 .*spdiagm(q, q+2,0 =>ones(T, q),1=> -2 .*ones(T, q), 2 =>ones(T, q))

    xy = Iy'*((Rr ⊗ (∂y ⊗ ∂x)))
    yy = Iy'*((Rr ⊗ (∂y² ⊗ Rp)) + kb^2 .*Iy)
    zy = Iy'*((∂z ⊗ (∂y ⊗ Rp)))

    #operators for Ez
    Rp = spdiagm(p+1, p+1, 0=>ones(T, p+1))
    Rq = spdiagm(q+1, q+1, 0=>ones(T, q+1))
    Rr = spdiagm(r, r+2, 1=>ones(T, r))

    Iz   = (Rr ⊗ (Rq ⊗ Rp))
    ∂x   = 1/Δx .*spdiagm(p+1, p+2, 0 =>-one(T) .*ones(T, p+1), 1 =>ones(T, p+1))
    ∂y   = 1/Δy .*spdiagm(q+1, q+2, 0 =>-one(T) .*ones(T, q+1), 1 =>ones(T, q+1))
    ∂z   = 1/Δz .*spdiagm(r, r+1, 0 =>-one(T) .*ones(T, r), 1 =>ones(T, r))
    ∂z²  = 1/(Δz)^2 .*spdiagm(r, r+2, 0 =>ones(T, r),1=> -2 .*ones(T, r), 2 =>ones(T, r))

    xz = Iz'*((∂z ⊗ (Rq ⊗ ∂x)))
    yz = Iz'*((∂z ⊗ (∂y ⊗ Rp)))
    zz = Iz'*((∂z² ⊗ (Rq ⊗ Rp)) + kb^2 .*Iz)
    
    return hcat(vcat(xx, xy, xz), vcat(yx, yy, yz), vcat(zx, zy, zz));
end

"""
Ig = _create_Greens_functions_restriction_operators(number_of_cells)\n
These operators are used to restrict the output of ∫GχEdV.\n
This integral is computed using an FFT on a domain that is larger the number of cells.\n
Therefore, the output of these should be restricted to the correct Yee cell grid.\n
"""
function _create_Greens_functions_restriction_operators(number_of_cells)
    T = ComplexF32;
    p, q, r = number_of_cells
    dx, dy, dz = 2 .^(ceil.(Int, log2.(number_of_cells.+2).+1))

    Ix = spdiagm(p+2, dx, 0=>ones(T, p+2))
    Iy = spdiagm(q+1, dy, 0=>ones(T, q+1))
    Iz = spdiagm(r+1, dz, 0=>ones(T, r+1))
    x = (Iz ⊗ (Iy ⊗ Ix))

    Ix = spdiagm(p+1, dx, 0=>ones(T, p+1))
    Iy = spdiagm(q+2, dy, 0=>ones(T, q+2))
    y = (Iz ⊗ (Iy ⊗ Ix))

    Iy = spdiagm(q+1, dy, 0=>ones(T, q+1))
    Iz = spdiagm(r+2, dz, 0=>ones(T, r+2))
    z = (Iz ⊗ (Iy ⊗ Ix))

    return blockdiag(x, y, z)
end

"""
Ig = _create_Greens_functions_restriction_operators_gpu(number_of_cells)\n
These operators are used to restrict the output of ∫GχEdV.\n
This integral is computed using an FFT on a domain that is larger the number of 2*cells.\n
Therefore, the output of these should be restricted to the correct Yee cell grid.\n
"""
function _create_Greens_functions_restriction_operators_gpu(number_of_cells)
    T = ComplexF32;
    p, q, r = number_of_cells
    dx, dy, dz = 2 .^(ceil.(Int, log2.(number_of_cells.+2).+1))

    Ix = spdiagm(p+2, dx, 0=>ones(T, p+2))
    Iy = spdiagm(q+1, dy, 0=>ones(T, q+1))
    Iz = spdiagm(r+1, dz, 0=>ones(T, r+1))
    x = (Iz ⊗ (Iy ⊗ Ix))

    Ix = spdiagm(p+1, dx, 0=>ones(T, p+1))
    Iy = spdiagm(q+2, dy, 0=>ones(T, q+2))
    y = (Iz ⊗ (Iy ⊗ Ix))

    Iy = spdiagm(q+1, dy, 0=>ones(T, q+1))
    Iz = spdiagm(r+2, dz, 0=>ones(T, r+2))
    z = (Iz ⊗ (Iy ⊗ Ix))

    Ig = blockdiag(x, y, z) |> CUDA.CUSPARSE.CuSparseMatrixCSC
    
    return Ig
end
