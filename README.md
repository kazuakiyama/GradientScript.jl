# GradientScript.jl
This script gives an example to compute loglikehood and its gradient (often called as a back projection, and 
proportional to the traditional residual map in CLEAN methods) from interferometric visibilities.

The script will use Comrade.jl to compute loglikelihood, and take a uvfits file to load visibilities.
The user optionally sets an input image for which loglikelihood will be evaluated.
Otherwise an empty image will be used for complex visibilities, or a Gaussian image with the unit total flux density
will be used for closure quantities.

## Julia version and setup
To use this we assume that you have Julia 1.10 installed. If you do not we recommend installing Julia with juliaup https://github.com/JuliaLang/juliaup. If you have juliaup install to use Julia 1.10 run the following commands
```bash
juliaup add lts
juliaup default lts
```
The main script to run is main.jl. To use it first call setup.jl:
```bash
julia setup.jl
```

## Usage
To compute the gradient, you can run the command
```bash
julia main.jl <uvfits> -o <path-to-output> ...
```
It will output the gradient image in a FITS file. `<uvfits>` indicates the path to the input uvfits for which the likelihood and its gradient to be computed. Other options are:
- `-o, --outpath`: the path to the output directory where images and other stats will be saved. Default is the current directory.
- `--fitsfile`: the input fits image, to which the gradient will be computed. 
- `--fovx`: the field of view in microarcseconds. Default is 200 μas.
- `--fovy`: the field of view in microarcseconds. Default is fovx.
- `-p, --psize`: the pixel size in microarcseconds. Default is 1 μas.
- `-x, --x`: the x offset of image center in microarcseconds. Default is 0 μas.
- `-y, --y`: the y offset of image center in microarcseconds. Default is 0 μas.
- `-u, --uvmin`: the minimum uv distance in λ. Default is 0.2e9.
- `--scanavg`: the average data at each scan at the loading
- `--ferr`: the fractional error in the data. Default is 0.0.
- `--closure`: use the closure quantities (lca, cp) instead of complex visibilities for the likelihood
- `--logimage`: use the logintensity for the image parameters
- `--comshift`: include the center-of-mass shift in the sky model