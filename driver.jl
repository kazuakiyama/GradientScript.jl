using Pkg;
Pkg.activate(@__DIR__);

using Pyehtim

using Comrade
using Distributions
using Enzyme
using LinearAlgebra
using VLBIImagePriors

function eval_likelihood(d::Comrade.AbstractVLBIPosterior, x::AbstractArray)
    return Comrade.loglikelihood(d, (; sky=(; x=x)))
end

function loglikelihood_and_gradient(d::Comrade.AbstractVLBIPosterior, x::AbstractArray)
    mode = Enzyme.EnzymeCore.WithPrimal(Comrade.admode(d))
    dx = zero(x)
    (_, y) = autodiff(mode, eval_likelihood, Active, Const(d), Duplicated(x, dx))
    return y, dx
end

