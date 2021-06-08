"""
constants = getConstants(freq).\n
freq is the frequency of interest of your simulation.\n
contants include:
c  : lightspeed\n
μ₀ : permeability of vacuum\n
ϵ₀ : permettivity of vacuum\n
f₀ : frequency in Hz\n
ω  : frequency in rad/s\n
kb : wavenumber of vacuum\n
λ  : wavelength in vacuum\n
"""
function getConstants(freq)
    c  = 299792458 #apparently this is what sim4life uses
    μ₀ = 4e-7*π
    ϵ₀ = 1/(c^2*μ₀)
    f₀ = freq #round is for the fact that i set the 3T and 7T freq to 128 and 298 MHz in sim4life
    ω  = 2*π*f₀
    kb = ω/c
    λ  = c/f₀
    return (;c, μ₀, ϵ₀, f₀, ω, kb, λ)
end