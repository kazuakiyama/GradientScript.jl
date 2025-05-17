function LinearImage(θ, metadata)
    (; x) = θ
    (; fluxnorm, comshift, grid) = metadata

    rast = reshape(x, size(grid)...)

    if fluxnorm
        rast = rast ./ sum(rast)
    end

    m = ContinuousImage(IntensityMap(rast, grid), BSplinePulse{3}())

    if comshift
        x0, y0 = centroid(m)
        return shifted(m, -x0, -y0)
    else
        return m
    end
end

function LogImage(θ, metadata)
    (; x) = θ
    (; fluxnorm, comshift, grid) = metadata

    rast = reshape(exp.(x), size(grid)...)

    # this is effectively using softmax transformation
    # which is an inverce of the centered logratio transform
    if fluxnorm
        rast = rast ./ sum(rast)
    end

    m = ContinuousImage(IntensityMap(rast, grid), BSplinePulse{3}())

    if comshift
        x0, y0 = centroid(m)
        return shifted(m, -x0, -y0)
    else
        return m
    end
end

