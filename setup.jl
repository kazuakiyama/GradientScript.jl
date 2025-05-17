using Pkg;
Pkg.activate(@__DIR__)
Pkg.instantiate()
Pkg.precompile()
include(joinpath(@__DIR__, "driver.jl"))
include(joinpath(@__DIR__, "skymodel.jl"))