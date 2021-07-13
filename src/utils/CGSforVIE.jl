"""
cgs_efield!(incident_electric_field, fft_tmp, G, vector_potential_3d, dielectric, vector_potential_1d, AToE, Ig, pfft, pifft, x, p, r, rt, u, v, q, uq; tol = 1e-15, maxit = 18)\n
Inplace version for solving the VIE. the result will be in x.\n
"""
function cgs_efield!(
    incident_electric_field,
    fft_tmp,
    G,
    vector_potential_3d,
    dielectric,
    vector_potential_1d,
    AToE,
    Ig,
    pfft,
    pifft,
    x,
    p,
    r,
    rt,
    u,
    v,
    q,
    uq;
    tol = 1e-15,
    maxit = 18
)
    T = ComplexF32
    n2b = norm(incident_electric_field)
    n2b = n2b*tol
    #copyto!(dst,src)
    copyto!(x, incident_electric_field)
    _total_minus_scattered_electric_field!(p, x, vector_potential_1d, vector_potential_3d, G, dielectric, Ig, AToE, fft_tmp, pfft, pifft, -one(T))
    # p = b - Ax
    axpy!(one(T), incident_electric_field, p)
    # r= b - Ax
    copyto!(r, p)
    # u = b - Ax
    copyto!(u, p)
    # v = A(b - Ax)
    _total_minus_scattered_electric_field!(v, p, vector_potential_1d, vector_potential_3d, G, dielectric, Ig, AToE, fft_tmp, pfft, pifft, one(T))
    copyto!(rt, r)
    ρ = dot(rt, r)
    for _ in 1:maxit
        σ = dot(rt, v)
        α = ρ/σ
        # q = u-α*v
        caxpy!(q, -α, v, u)
        # uq = u + q
        caxpy!(uq, one(T), u, q)
        # x = x + α*uq
        axpy!(α, uq, x)

        _total_minus_scattered_electric_field_for_residual!(r, uq, vector_potential_1d, vector_potential_3d, G, dielectric, Ig, AToE, fft_tmp, pfft, pifft, -α)

        if norm(r) <= n2b
          break #return xx,xy,xz
        end

        ρold = ρ
        ρ = dot(rt, r)
        β = ρ/ρold
        # u = r + β*q
        caxpy!(u, β, q, r)
        #p = u + β*(q+β*p)
        axpby!(one(T), q, β, p)
        axpby!(one(T), u, β, p)
        _total_minus_scattered_electric_field!(v, p, vector_potential_1d, vector_potential_3d, G, dielectric, Ig, AToE, fft_tmp, pfft, pifft, one(T))
    end

    return nothing
end
