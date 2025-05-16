using Pkg;
Pkg.activate(@__DIR__);

using Comonicon

include(joinpath(@__DIR__, "modules.jl"))
include(joinpath(@__DIR__, "skymodel.jl"))

LinearAlgebra.BLAS.set_num_threads(1)
if Threads.nthreads() > 1
    VLBISkyModels.NFFT._use_threads[] = false
    VLBISkyModels.FFTW.set_num_threads(Threads.nthreads())
end

@main function main(
    uvfile::String;
    outpath::String="",
    array::String="",
    fovx::Float64=200.0, fovy::Float64=fovx,
    psize::Float64=1.0,
    x::Float64=0.0, y::Float64=0.0,
    uvmin::Float64=0e9,
    benchmark::Bool=false,
    scanavg::Bool=false,
    ferr::Float64=0.0,
    closure::Bool=false,
    logimage::Bool=false,
    comshift::Bool=false,
)
    # field of view in rad
    fovxrad = μas2rad(fovx)
    fovyrad = μas2rad(fovy)

    # number of pixels in the image
    nx = ceil(Int, fovx / psize)
    ny = ceil(Int, fovy / psize)

    # output directory
    outpath = isempty(outpath) ? first(splitext(uvfile)) : joinpath(outpath, first(splitext(basename(uvfile))))

    @info "Fitting the data: $uvfile"
    @info "Loading the array file: $array"
    @info "Outputing to $outpath"
    @info "Field of view: ($fovx, $fovy) μas"
    @info "number of pixels: ($nx, $ny)"
    @info "Image center offset: ($x, $y) μas"

    # load the uv data
    obs = ehtim.obsdata.load_uvfits(uvfile)
    if scanavg
        @info "Averaging the data over scans"
        obsavg = scan_average(obs.flag_uvdist(uv_min=uvmin))
    else
        obsavg = obs.flag_uvdist(uv_min=uvmin)
    end
    @info "Adding $ferr fractional error to the data"
    if ferr > 0
        obsavg = obsavg.add_fractional_noise(ferr)
    end
    dvis = extract_table(obsavg, Visibilities())
    beam = beamsize(dvis)

    # define the image grid
    x0 = μas2rad(x)
    y0 = μas2rad(y)
    hdr = ComradeBase.MinimalHeader(string(dvis.config.source),
        dvis.config.ra, dvis.config.dec,
        dvis.config.mjd, dvis[:baseline].Fr[1] # assume all frequencies are the same
    )
    if Threads.nthreads() > 1
        g = imagepixels(fovxrad, fovyrad, nx, ny, x0, y0; executor=ThreadsEx(:dynamic), header=hdr)
    else
        g = imagepixels(fovxrad, fovyrad, nx, ny, x0, y0; header=hdr)
    end
    @info "Beam relative to pixel size: = $(beam/μas2rad(psize))"

    # define the data terms
    if closure
        @info "Using closure amplitudes and phases to compute the back projection"
        dlca, dcp = extract_table(
            obsavg,
            LogClosureAmplitudes(; snrcut=3),
            ClosurePhases(; snrcut=3)
        )
        dtabs = (; lca=dlca, cp=dcp)
        skymeta = (; fluxnorm=true, comshift=comshift, grid=g)
    else
        @info "Using complex visibilities to compute the back projection"
        dtabs = (; vis=dvis)
        skymeta = (; fluxnorm=false, comshift=false, grid=g)
    end

    # define the image function for the sky model
    if logimage
        @info "Using log transform for the image function"
        skyfunc = LogImage
    else
        @info "Using linear image function"
        skyfunc = LinearImage
    end

    # Define the image prior
    #   we don't include a prior (denoiser), as this is supposed to be done by R2D2.
    d = Normal(0, 1)
    #d = Uniform(-Inf, Inf)
    prior = (x=Tuple(fill(d, nx * ny)),)

    # sky model
    skym = SkyModel(skyfunc, prior, g; metadata=skymeta)

    # define the posterior
    @info "Defining posterior"
    post = VLBIPosterior(
        skym, dtabs...;
        admode=set_runtime_activity(Enzyme.Reverse)
    )
    @info "Posterior defined"
    tpost = asflat(post)
    ndim = dimension(tpost)
    @info ndim

    # initial image
    x0 = zeros(ndim)
    lpdf = Comrade.logdensityof(tpost, x0)
    @info "LogPDF: $lpdf"

    # gradient image
    @info "Computing the gradient image:"
    outs = Comrade.LogDensityProblems.logdensity_and_gradient(tpost, x0)
    gradimg = IntensityMap(reshape(outs.gradient, size(g)...), g)

    # output results
    @info "Outputing to $outpath"
    mkpath(dirname(outpath))
    imgout = joinpath(mkpath(joinpath(dirname(outpath), "images")), basename(outpath) * "_results")
    Comrade.save_fits(imgout * "_gradient.fits", gradimg)
end