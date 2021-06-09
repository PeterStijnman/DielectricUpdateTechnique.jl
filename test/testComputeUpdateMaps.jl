import DielectricUpdateTechique as DUTCH
using Test

x = collect(0:1f-3:2f-3)
y = collect(0:1f-3:2f-3)
z = collect(0:1f-3:2f-3)

vσ = rand(Float32,3, 3, 3);
vϵ = rand(Float32,3, 3, 3);

bg = DUTCH.cellToYeeDielectric(vσ, vϵ, x, y, z)

vσ[1:2,2,2] = 2 .* vσ[1:2,2,2];

imp = DUTCH.cellToYeeDielectric(vσ, vϵ, x, y, z)

Δ, loc = DUTCH.computeUpdateMaps(bg,imp)

@test round(Δ[:σˣ][2,2,2], digits=3) == round(imp[:σˣ][2,2,2] - bg[:σˣ][2,2,2], digits=3)
@test round(Δ[:σˣ][1,2,2], digits=3) == round(imp[:σˣ][1,2,2] - bg[:σˣ][1,2,2], digits=3)

@test round(Δ[:σʸ][2,2,2], digits=3) == round(imp[:σʸ][2,2,2] - bg[:σʸ][2,2,2], digits=3)
@test round(Δ[:σʸ][1,2,2], digits=3) == round(imp[:σʸ][1,2,2] - bg[:σʸ][1,2,2], digits=3)

@test round(Δ[:σᶻ][2,2,2], digits=3) == round(imp[:σᶻ][2,2,2] - bg[:σᶻ][2,2,2], digits=3)
@test round(Δ[:σᶻ][1,2,2], digits=3) == round(imp[:σᶻ][1,2,2] - bg[:σᶻ][1,2,2], digits=3)

# there are 20 edges to be updated for 2 neighbouring cells
@test sum(vec(loc[:updateLocˣ])) + sum(vec(loc[:updateLocʸ])) + sum(vec(loc[:updateLocᶻ])) == 20


vσ = rand(Float32,3, 3, 3);
vϵ = rand(Float32,3, 3, 3);

bg = DUTCH.cellToYeeDielectric(vσ, vϵ, x, y, z)

vσ[1,2,2] = 2 .* vσ[1,2,2];
vσ[3,2,2] = 2 .* vσ[3,2,2];

imp = DUTCH.cellToYeeDielectric(vσ, vϵ, x, y, z)

Δ, loc = DUTCH.computeUpdateMaps(bg,imp)
# there are 24 edges to be updated for 2 non neighbouring cells
@test sum(vec(loc[:updateLocˣ])) + sum(vec(loc[:updateLocʸ])) + sum(vec(loc[:updateLocᶻ])) == 24

# test for getS(_gpu) and getC 

x = collect(0:1f-3:5f-3)
y = collect(0:1f-3:5f-3)
z = collect(0:1f-3:5f-3)


vσ = rand(Float32,5, 5, 5);
vϵ = rand(Float32,5, 5, 5);

bg = DUTCH.cellToYeeDielectric(vσ, vϵ, x, y, z)

vσ[2,2,2] = 2 .* vσ[2,2,2];
vσ[4,2,2] = 2 .* vσ[4,2,2];

imp = DUTCH.cellToYeeDielectric(vσ, vϵ, x, y, z)

Δ, loc = DUTCH.computeUpdateMaps(bg,imp)

S = DUTCH.getS(loc)

@test count(S.N .==1) == 24
@test size(S.N) == ((length(x)-1)*(length(y)-2)*(length(z)-2) + (length(x)-2)*(length(y)-1)*(length(z)-2) + (length(x)-2)*(length(y)-2)*(length(z)-1), 24)
@test count(S.T .==1) == 24
@test size(S.T) == (24, (length(x)-1)*(length(y)-2)*(length(z)-2) + (length(x)-2)*(length(y)-1)*(length(z)-2) + (length(x)-2)*(length(y)-2)*(length(z)-1))

S = DUTCH.getS_gpu(loc) |> x -> (N = collect(x.N), T = collect(x.T))

@test count(S.N .==1) == 24
@test size(S.N) == ((length(x)-1)*(length(y)-2)*(length(z)-2) + (length(x)-2)*(length(y)-1)*(length(z)-2) + (length(x)-2)*(length(y)-2)*(length(z)-1), 24)
@test count(S.T .==1) == 24
@test size(S.T) == (24, (length(x)-1)*(length(y)-2)*(length(z)-2) + (length(x)-2)*(length(y)-1)*(length(z)-2) + (length(x)-2)*(length(y)-2)*(length(z)-1))

f = 300e6; #300MHz
m = DUTCH.getConstants(f);
C = DUTCH.getC(Δ,m)

@test length(C) == 24
@test sum(imag.(C)) == 0f0
@test round(sum(real.(C)),digits=1)  == round(sum(real.(Δ.σˣ)) + sum(real.(Δ.σʸ)) + sum(real.(Δ.σᶻ)),digits = 1)
@test typeof(C) == Vector{ComplexF32}


