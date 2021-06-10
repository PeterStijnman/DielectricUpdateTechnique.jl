import DielectricUpdateTechique as DUTCH
using Test

nCells = [30,33,61];
res = [1f-3,1f-3,1f-3];

r = DUTCH.createGridGreensFunction(nCells,res);

@test size(r) == (64,128,128)
@test r[33,65,65] == 0
@test r[34,66,66] == sqrt(res[1]^2 + res[2]^2 + res[3]^2)
@test r[33,66,66] == sqrt(res[2]^2 + res[3]^2)
@test r[1,66,66] == sqrt((32*res[1])^2 + res[2]^2 + res[3]^2)

G = DUTCH._createGreensFunction(r,6.4,res[1]/2f0);

@test size(G) == size(r)
@test maximum(abs.(G)) == abs(G[1,1,1])

Gdv = DUTCH.createGreensFunctions(nCells,res,6.4);

@test Gdv[1,1,1] == G[1,1,1]*prod(res)
