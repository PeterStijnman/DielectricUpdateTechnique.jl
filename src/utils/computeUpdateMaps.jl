"""
Δdielectric, update_locations = get_update_maps(background_dielectric, desired_dielectric)\n 
Computes the difference between two dielectric maps.\n 
Useful when comparing with other simulation techniques like FDTD.\n
From the resulting outcomes the mapping matrix S and the dielectric update vector C can be computed.\n
The inputs can be obtained from:\n 
\n 
background_dielectric = cell_to_yee_dielectric(vσ, vϵ, x, y, z)\n 
desired_dielectric = cell_to_yee_dielectric(vσ, vϵ, x, y, z)\n 
\n
Where the vσ and vϵ are the conductivity and permittivity for the different simulations.\n
"""
function get_update_maps(background_dielectric::NamedTuple, desired_dielectric::NamedTuple)
    dimensionsˣ = size(desired_dielectric[:σˣ])
    dimensionsʸ = size(desired_dielectric[:σʸ])
    dimensionsᶻ = size(desired_dielectric[:σᶻ])
    
    # compute the update dielectric
    σˣ = round.(desired_dielectric[:σˣ] - background_dielectric[:σˣ],digits = 3)
    σʸ = round.(desired_dielectric[:σʸ] - background_dielectric[:σʸ],digits = 3)
    σᶻ = round.(desired_dielectric[:σᶻ] - background_dielectric[:σᶻ],digits = 3)
    ϵˣ = round.(desired_dielectric[:ϵˣ] - background_dielectric[:ϵˣ],digits = 3)
    ϵʸ = round.(desired_dielectric[:ϵʸ] - background_dielectric[:ϵʸ],digits = 3)
    ϵᶻ = round.(desired_dielectric[:ϵᶻ] - background_dielectric[:ϵᶻ],digits = 3)

    #alloc update map
    updateLocˣ = zeros(Bool, dimensionsˣ...)
    updateLocʸ = zeros(Bool, dimensionsʸ...)
    updateLocᶻ = zeros(Bool, dimensionsᶻ...)

    #compute the update maps
    tmp = findall(x -> abs(x) > 0.0, σˣ)
    updateLocˣ[tmp] .= true
    tmp = findall(x -> abs(x) > 0.0, ϵˣ)
    updateLocˣ[tmp] .= true

    tmp = findall(x -> abs(x) > 0.0, σʸ)
    updateLocʸ[tmp] .= true
    tmp = findall(x -> abs(x) > 0.0, ϵʸ)
    updateLocʸ[tmp] .= true

    tmp = findall(x -> abs(x) > 0.0, σᶻ)
    updateLocᶻ[tmp] .= true
    tmp = findall(x -> abs(x) > 0.0, ϵᶻ)
    updateLocᶻ[tmp] .= true

    return (;σˣ, ϵˣ, σʸ, ϵʸ, σᶻ, ϵᶻ),(;updateLocˣ, updateLocʸ, updateLocᶻ)
end