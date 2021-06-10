import DielectricUpdateTechique as DUTCH
using Test

nCells = [5,12,10];
nUpdates = [4,4,4]
jv,eI,a,A,efft,pfft,pifft = DUTCH.allocateSpaceVIE(nCells,nUpdates)

tl = (nCells[1]+2)*(nCells[2]+1)*(nCells[3]+1) + (nCells[1]+1)*(nCells[2]+2)*(nCells[3]+1) + (nCells[1]+1)*(nCells[2]+1)*(nCells[3]+2)

@test length(jv) == sum(nUpdates)
@test abs(sum(jv)) == 0
@test length(eI) == tl
@test abs(sum(eI)) == 0
@test length(a) == tl
@test abs(sum(a)) == 0
@test length(A) == 16*32*32*3
@test abs(sum(A)) == 0
@test length(efft) == 16*32*32*3
@test abs(sum(efft)) == 0

x,p,r,rt,u,v,q,uq = DUTCH.allocateCGSVIE(nCells)

@test length(x) == tl
@test abs(sum(x)) == 0
@test length(p) == tl
@test abs(sum(p)) == 0
@test length(r) == tl
@test abs(sum(r)) == 0
@test length(rt) == tl
@test abs(sum(rt)) == 0
@test length(u) == tl
@test abs(sum(u)) == 0
@test length(v) == tl
@test abs(sum(v)) == 0
@test length(q) == tl
@test abs(sum(q)) == 0
@test length(uq) == tl
@test abs(sum(uq)) == 0
