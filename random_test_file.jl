using Revise
using BenchmarkTools
import DielectricUpdateTechique as DUTCH


frequency = 300f6;
np, nq, nr = [25,25,25];
hp, hq, hr = div.([np, nq, nr],2);

# create grid for the problem
res = [1f-3,1f-3,1f-3];
x_axis = -hp*res[1]:res[1]:hp*res[1] |> collect |> s -> repeat(s,outer=(1,nq,nr));
y_axis = -hq*res[2]:res[2]:hq*res[2] |> collect |> s -> reshape(s,(1,:))|> s -> repeat(s,outer=(np,1,nr));
z_axis = -hr*res[3]:res[3]:hr*res[3] |> collect |> s -> reshape(s,(1,1,:))|> s -> repeat(s,outer=(np,nq,1));

# create background dielectric (sphere within vacuum)
BG_σ = zeros(Float32,np,nq,nr);
BG_ϵr = ones(Float32,np,nq,nr);

# create the cube with σ = 0.15 S/m and ϵr = 40
BG_σ[hp-5:hp+5,hq-5:hq+5,hr-5:hr+5] .= 0.15;
BG_ϵr[hp-5:hp+5,hq-5:hq+5,hr-5:hr+5] .= 40;

# create the scenario we want to obtain 
# a smaller sphere within the background dielectric
Update_σ = copy(BG_σ);
Update_ϵr = copy(BG_ϵr);

# the plate next to the cube that has σ = 5 S/m and ϵr = 10
Update_σ[hp+6:hp+8,hq-6:hq+6,hr-6:hr+6] .= 3;
Update_ϵr[hp+6:hp+8,hq-6:hq+6,hr-6:hr+6] .= 20;

# define the source to calculate the incidenct electric field
source = zeros(ComplexF32,np*(nq-1)*(nr-1)+(np-1)*nq*(nr-1)+(np-1)*(nq-1)*nr);
source[7201] = 1f0 + 0f0im;
source[21601] = 0f0 + 1f0im;
source[35713] = 1f-1 + 3f-1im;

# calculate the incident and total electric field starting from vacuum surrounding.
Einc = DUTCH.calculate_electric_field_vacuum_to_dielectric(BG_σ,BG_ϵr,res,source,frequency);
Etot = DUTCH.calculate_electric_field_vacuum_to_dielectric(Update_σ,Update_ϵr,res,source,frequency);

prob = DUTCH.create_problem(BG_σ,BG_ϵr,Update_σ,Update_ϵr,Einc,frequency,res);
Jsc,Esc = DUTCH.solve_problem(prob,tol=1f-4);
Etot_update = Einc + Esc;

err = DUTCH.norm(Etot-Einc-Esc);
norm_Esc = DUTCH.norm(Esc);
@test err/norm_Esc < 0.07