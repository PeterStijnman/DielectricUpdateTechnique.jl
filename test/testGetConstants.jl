import DielectricUpdateTechique as DUTCH
using Test

f = 300f6; #300MHz
m = DUTCH.get_constants(f);

pif32 = π |> Float32;

@test m[:c]  == 299792458
@test m[:μ₀] == 4f-7*pif32
@test m[:ϵ₀] == 1/(m[:c]^2*m[:μ₀])
@test m[:f₀] == f
@test m[:ω]  == 2*pif32*f
@test m[:kb] == 2*pif32*f/m[:c]
@test m[:λ]  == m[:c]/f

f = 1337f6; #1337MHz
m = DUTCH.get_constants(f);
@test m[:f₀] == f
@test m[:ω]  == 2*pif32*f
@test m[:kb] == 2*pif32*f/m[:c]
@test m[:λ]  == m[:c]/f
