# GemmDemo.jl
A pure-Julia, BLIS-style* dgemm demo.

![performance comparison image](https://github.com/Sacha0/GemmDemo.jl/blob/master/perfvis.png)

Performance comparison of dgemm implementations (specifically `C = AB + C`, no `α` or `β`). The horizontal axis provides matrix size/shape; all matrices are square and of the same size. The vertical axis provides average GFLOPS achieved over the operation (with an i5-7360U, i.e. laptop Kaby Lake).

The solid black line is OpenBLAS. The dotted black line is a simple three-nested-loop implementation. The dashed orange line is a pure Julia implemention, BLIS-style* but without cache block packing. The dashed purple line is the same, but with cache block packing. The solid blue line dispatches between the former two implementations dependent on the problem (details below**).

The packing implementation stays within ~10% of OpenBLAS performance.

The code overall has seen essentially no optimization effort beyond hitting the structural highlights: Nothing is manually prefetched. Nothing is manually unrolled. The microkernel is otherwise suboptimal. The cache parameters are suboptimal. The packing implementation is naive. This performance represents a 20%-effort-80% result point. In other words, there remain quite a few opportunities to whittle away at that ~10% asymptotic performance gap to OpenBLAS.

Find the demo code in [src.jl](https://github.com/Sacha0/GemmDemo.jl/blob/master/src.jl).

The code is fully parameterized (cache parameters, microkernel shape) apart from the microkernels, which are simple enough that macro-generating them from their parameters (microkernel shape and supported vector extensions) should be straightforward (given implementation bandwidth). In other words, programmatically generating variations on this code for different hardware-supported data types and architectures should be possible, and likewise for non-hardware types but with a generic microkernel (that can be replaced with something optimized if desired).

Cleanup loops not implemented yet, so test matrices must be multiples of the microkernel size.

(* For more on BLIS, check out the absolutely wonderful work of the Science of High Performance Computing (SHPC) group at UT Austin: http://www.cs.utexas.edu/~flame/web/ and http://shpc.ices.utexas.edu.)

(** When the problem entirely/mostly fits into reasonably fast cache (e.g. L3), packing overhead tends to outweigh packing's benefits; when the problem spills beyond that fast cache, packing's benefits tend to outweigh packing overhead. That the non-packing implementation outperforms the packing implementation for small problems and conversely for large reflects this point. I had a hunch that, on this and similar architectures, a reasonable heuristic for choosing whether or not to pack would be whether the full problem fits into L3 cache; consequently the solid blue line dispatches the non-packing implementation in the former case and the packing implementation otherwise. Apparently that was a good hunch, as it looks like L3 cache size is roughly the crossover point for the packing and non-packing implementations on this machine.)

