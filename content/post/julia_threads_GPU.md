+++
title = "Julia performance - Threads &  GPU"
date = "2021-03-25"
description = "How to improve Julia performance by multithreading and GPU compute."
tags = [
    "Julia",
    "threads",
    "GPU",
    "AFM",
    "atomic force microscopy",
    "Probe particle model",
    "grid"
]
+++

This article gives real-life examples of how to improve performance by using multithreaded execution and GPU compute.


## First, some scientific background

I have recently played around with scientific simulations for bond-resolved atomic force microscopy (AFM).
If you do not know what this is, just a one-liner explanation: local force measurements can be used to image the bond-structure of single molecules.
More information can be found in the [seminal paper by Leo Gross et al.](https://science.sciencemag.org/content/325/5944/1110.abstract)

Now the whole imaging mechanism is a little complicated, but very useful models have been developed to simulate and analyze such images.
One mile stone was the introduction of the **Probe Particle Model**, developed by Prokop Hapala, Pavel Jelinek et al.,
see [here](https://journals.aps.org/prb/abstract/10.1103/PhysRevB.90.085421) and [here](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.113.226101).

![AFM simulation of an olympicene-like molecule](/Tour-de-Code/images/julia_threads_GPU_df_olympicene.png
 "Probe Particle Model simulation of an AFM image of a single olympicene-like molecule (image width: 1.6 nm)")

In this model a force field is calculated
([Lennard Jones](https://en.wikipedia.org/wiki/Lennard-Jones_potential) and [Coulomb forces](https://en.wikipedia.org/wiki/Coulomb%27s_law))
and the probe particle (which is attached to the atomic force microscopy tip) is _relaxed_
within this force field, i.e. it can respond to the forces and shift out of its starting position.
The [original code](https://github.com/ProkopHapala/ProbeParticleModel) that has been made open source (highly appreciated!)
is programmed in Python and C.


## Ok, so finally we talk about code

I have implemented some basic parts of it in Julia and want to talk a little bit about performance.
The original code runs relatively slow for a 100x100x100 grid of points - so the first most obvious thing was to use multi-threading.
This is actually very easy to add!

In the simple code there is some main loop that looks like this:

```julia
for i in eachindex(grid)
    # computation for each point in the grid
    # ...
end
```

All I had to do was to add the `@Threads.threads` macro:
```julia
@Threads.threads for i in eachindex(grid)
    # computation for each point in the grid
    # ...
end
```

That's it! Now just run julia with [threading support](https://docs.julialang.org/en/v1/manual/multi-threading/): `julia --threads 4`.

On my old rusty i5-3570K with four threads this approach will lead to a speedup **by a factor of 3.5!** Isn't that awesome?


## The grand finale: GPUs for the win!

The loop above can also be written in one function that broadcasts over `grid`. You will see in a moment why this is a convenient way for GPU computation.

```julia
forcefield = force_at_point.(grid, (parameter1, ), (parameter2, ))
```

The brackets around `parameter1` and `parameter2` are there to avoid broadcasting over those variables.

But now let's get to the pralellization on the GPU. We will make use of [CUDA arrays](https://github.com/JuliaGPU/CUDA.jl) that work with NVIDIA cards - I have a NVIDIA GeForce GTX 1080
in my PC.

{{< highlight julia >}}
using CUDA
grid_cu = cu(grid)  # creates a CUDA array
forcefield = force_at_point.(grid_cu, (parameter1, ), (parameter2, ))
{{< /highlight >}}

Broadcasting will automatically work on CUDA arrays. It can be as easy as that! In some cases you might want to write a [GPU kernel function](https://juliagpu.github.io/CUDA.jl/dev/tutorials/introduction/).
OK, so how much faster did we get? A whopping **50-fold increase in performance** compared to the single-threaded calculation.
<br /><br />