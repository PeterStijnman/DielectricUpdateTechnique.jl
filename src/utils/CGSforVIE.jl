"""
cgs_efield!(eI,efft,G,A,χ,a,AToE,Ig,pfft,pifft,x,p,r,rt,u,v,q,uq;tol = 1e-15,maxit = 18)\n
Inplace version for solving the VIE. the result will be in x.\n
"""
function cgs_efield!(eI,efft,G,A,χ,a,AToE,Ig,pfft,pifft,x,p,r,rt,u,v,q,uq;tol = 1e-15,maxit = 18)
  T = ComplexF32
  n2b = norm(eI)
  n2b = n2b*tol
  #copyto!(dst,src)
  copyto!(x,eI)
  ETotalMinEScattered!(p,x,a,A,G,χ,Ig,AToE,efft,pfft,pifft,-one(T))
  # p = b - Ax
  axpy!(one(T),eI,p)
  # r= b - Ax
  copyto!(r,p)
  # u = b - Ax
  copyto!(u,p)
  # v = A(b - Ax)
  ETotalMinEScattered!(v,p,a,A,G,χ,Ig,AToE,efft,pfft,pifft,one(T))
  copyto!(rt,r)
  ρ = dot(rt,r)
  for _ in 1:maxit
    σ = dot(rt,v)
    α = ρ/σ
    # q = u-α*v
    caxpy!(q,-α,v,u)
    # uq = u + q
    caxpy!(uq,one(T),u,q)
    # x = x + α*uq
    axpy!(α,uq,x)

    ETotalMinEScatteredForResidual!(r,uq,a,A,G,χ,Ig,AToE,efft,pfft,pifft,-α)

    if norm(r) <= n2b
      break #return xx,xy,xz
    end

    ρold = ρ
    ρ = dot(rt,r)
    β = ρ/ρold
    # u = r + β*q
    caxpy!(u,β,q,r)
    #p = u + β*(q+β*p)
    axpby!(one(T),q,β,p)
    axpby!(one(T),u,β,p)
    ETotalMinEScattered!(v,p,a,A,G,χ,Ig,AToE,efft,pfft,pifft,one(T))
  end
end
