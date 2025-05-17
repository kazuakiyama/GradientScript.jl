using Pkg;
Pkg.activate(@__DIR__);

using Comonicon

include(joinpath(@__DIR__, "driver.jl"))
include(joinpath(@__DIR__, "skymodel.jl"))

LinearAlgebra.BLAS.set_num_threads(1)
if Threads.nthreads() > 1
    VLBISkyModels.NFFT._use_threads[] = false
    VLBISkyModels.FFTW.set_num_threads(Threads.nthreads())
end

@main function main(
    uvfile::String;
    outpath::String="",
    fitsfile::String="",
    fovx::Float64=200.0, fovy::Float64=fovx,
    psize::Float64=1.0,
    x::Float64=0.0, y::Float64=0.0,
    uvmin::Float64=0e9,
    scanavg::Bool=false,
    ferr::Float64=0.0,
    closure::Bool=false,
    logimage::Bool=false,
    comshift::Bool=false,
)
    if isempty(fitsfile)
        @info "No fits file provided. Using fovx/fovy and nx/ny to define the image grid"
        # field of view in rad
        fovxrad = μas2rad(fovx)
        fovyrad = μas2rad(fovy)

        # number of pixels in the image
        nx = ceil(Int, fovx / psize)
        ny = ceil(Int, fovy / psize)
    else
        @info "FITS file provided. Using it to define the image grid"
        # load image
        imfits = Comrade.load_fits(fitsfile, IntensityMap)

        # get the pixel information
        nx, ny = size(imfits)
        fovxrad = abs(imfits.X[2] - imfits.X[1]) * nx
        fovyrad = abs(imfits.Y[2] - imfits.Y[1]) * ny

        # get the field of view in μas
        fovx = rad2μas(fovxrad)
        fovy = rad2μas(fovyrad)

        # get x, y in μas
        x = rad2μas(imfits.X[end] - imfits.X[1]) / 2
        y = rad2μas(imfits.Y[end] - imfits.Y[1]) / 2
    end

    # output directory
    outpath = isempty(outpath) ? first(splitext(uvfile)) : joinpath(outpath, first(splitext(basename(uvfile))))

    @info "Fitting the data: $uvfile"
    @info "Outputing to $outpath"
    @info "Field of view: ($fovx, $fovy) μas"
    @info "number of pixels: ($nx, $ny)"
    @info "Image center offset: ($x, $y) μas"
    @info "Adding $ferr fractional error to the data"

    # load the uv data
    obs = ehtim.obsdata.load_uvfits(uvfile)
    if scanavg
        @info "Averaging the data over scans"
        obsavg = scan_average(obs.flag_uvdist(uv_min=uvmin))
    else
        obsavg = obs.flag_uvdist(uv_min=uvmin)
    end
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
        @info "Using log closure amplitudes and closure phases to compute the back projection"
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
    prior = (x=MvNormal(ones(ny * ny)),)

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
    if isempty(fitsfile) == false
        @info "Using the provided image for the initial image"
        x0[1:ndim] = imfits[1:ndim]
        if skymeta.fluxnorm
            if logimage
                # This particular example assumes the fits file stores log intensity
                linx0 = exp.(x0)
                linx0 ./= sum(linx0)
                x0 = log.(linx0)
            else
                x0[1:ndim] = imfits[1:ndim]
                x0 ./= sum(imfits)
            end
        end
    elseif closure
        @info "Using a Gaussian prior for the initial image"
        initimage = intensitymap(modify(Gaussian(), Stretch(fovxrad, fovyrad)), g)
        x0[1:ndim] = initimage[1:ndim]
        if logimage
            x0 = log.(x0)
        end
    else
        @info "Using an empty image for the initial image"
    end

    # evaluate the log posterior
    @info "Computing the log likelihood and its gradient"
    loglh, dloglh = loglikelihood_and_gradient(post, x0)
    @info "LogLikelihood: $(loglh)"

    # convert the gradient to an intensitymap object
    gradimg = IntensityMap(reshape(dloglh, size(g)...), g)

    # output results
    @info "Outputing to $outpath"
    mkpath(dirname(outpath))
    imgout = joinpath(mkpath(joinpath(dirname(outpath), "images")), basename(outpath) * "_results")
    Comrade.save_fits(imgout * "_gradient.fits", gradimg)
end