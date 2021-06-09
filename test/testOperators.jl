import DielectricUpdateTechique as DUTCH
using Test

nCells = [25,54,37];
p,q,r = nCells
resolution = [1f-3,1f-3,1f-3];
m = DUTCH.getConstants(300f6);
k_b = m.kb

AToE = DUTCH.createSparseDifferenceOperators(nCells,resolution,k_b);

xl = (p+2)*(q+1)*(r+1)
yl = (p+1)*(q+2)*(r+1)
zl = (p+1)*(q+1)*(r+2)

@test size(AToE) == (xl+yl+zl, xl+yl+zl)
@test real(AToE[2,2]) == -2/resolution[1]^2 + k_b^2
@test real(AToE[3,2]) == 1/resolution[1]*1/resolution[2]

Ig = DUTCH.createGreensFunctionsRestrictionOperators(nCells);
dx,dy,dz = 2 .^(ceil.(Int,log2.(nCells.+2).+1))

@test size(Ig.N) == (xl+yl+zl, dx*dy*dz*3)
@test size(Ig.T) == (dx*dy*dz*3, xl+yl+zl)
