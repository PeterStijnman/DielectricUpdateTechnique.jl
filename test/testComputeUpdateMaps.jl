import DielectricUpdateTechique as DUTCH
using Test

x = collect(0:1f-3:2f-3)
y = collect(0:1f-3:2f-3)
z = collect(0:1f-3:2f-3)

vσ = rand(Float32,3, 3, 3);
vϵ = rand(Float32,3, 3, 3);

bg = DUTCH.cell_to_yee_dielectric(vσ, vϵ, x, y, z)

vσ[1:2,2,2] = 2 .* vσ[1:2,2,2];

imp = DUTCH.cell_to_yee_dielectric(vσ, vϵ, x, y, z)

Δ, loc = DUTCH.get_update_maps(bg,imp)

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

bg = DUTCH.cell_to_yee_dielectric(vσ, vϵ, x, y, z)

vσ[1,2,2] = 2 .* vσ[1,2,2];
vσ[3,2,2] = 2 .* vσ[3,2,2];

imp = DUTCH.cell_to_yee_dielectric(vσ, vϵ, x, y, z)

Δ, loc = DUTCH.get_update_maps(bg,imp)
# there are 24 edges to be updated for 2 non neighbouring cells
@test sum(vec(loc[:updateLocˣ])) + sum(vec(loc[:updateLocʸ])) + sum(vec(loc[:updateLocᶻ])) == 24

# test for get_S_matrix(_gpu) and get_C_matrix 

x = collect(0:1f-3:5f-3)
y = collect(0:1f-3:5f-3)
z = collect(0:1f-3:5f-3)


vσ = rand(Float32,5, 5, 5);
vϵ = rand(Float32,5, 5, 5);

bg = DUTCH.cell_to_yee_dielectric(vσ, vϵ, x, y, z)

vσ[2,2,2] = 2 .* vσ[2,2,2];
vσ[4,2,2] = 2 .* vσ[4,2,2];

imp = DUTCH.cell_to_yee_dielectric(vσ, vϵ, x, y, z)

Δ, loc = DUTCH.get_update_maps(bg,imp)

S = DUTCH.get_S_matrix(loc)

@test count(S .== 1) == 24
@test size(S) == ((length(x)-1)*(length(y)-2)*(length(z)-2) + (length(x)-2)*(length(y)-1)*(length(z)-2) + (length(x)-2)*(length(y)-2)*(length(z)-1), 24)

#=
S = DUTCH.get_S_matrix_gpu(loc) |> collect

@test count(S .== 1) == 24
@test size(S) == ((length(x)-1)*(length(y)-2)*(length(z)-2) + (length(x)-2)*(length(y)-1)*(length(z)-2) + (length(x)-2)*(length(y)-2)*(length(z)-1), 24)
=#

@test sum(DUTCH._get_number_of_updated_edges(loc)) == 24

f = 300e6; #300MHz
m = DUTCH.get_constants(f);
C = DUTCH.get_C_matrix(Δ,m)

@test length(C) == 24
@test sum(imag.(C)) == 0
@test round(sum(real.(C)),digits=1)  == round(sum(real.(Δ.σˣ)) + sum(real.(Δ.σʸ)) + sum(real.(Δ.σᶻ)),digits = 1)
@test typeof(C) == Vector{ComplexF32}


χ = DUTCH._set_dielectric(bg,m,[3,3,3]);
@test abs.(χ.vac) == 1
@test size(χ.patient) == (16,16,16,3)
@test abs.(χ.patient[8,7,7,1]) == 0
@test abs.(χ.patient[3,3,3,1]) != 0
@test abs.(χ.patient[7,8,7,2]) == 0
@test abs.(χ.patient[3,3,3,2]) != 0
@test abs.(χ.patient[7,7,8,3]) == 0
@test abs.(χ.patient[3,3,3,3]) != 0