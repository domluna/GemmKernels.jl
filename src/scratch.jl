# Notes:
#
# groups of 32 threads seem to be the standard choice (called warps)
#
# WMMA.Config{M, N, K, d_type}
#
# WMMA instructions calculate the matrix multiply-accumulate 
# operation  D = A * B + C
#
# A is MxK
# B is KxN
# C, D are MxN
#
# where MxK, KxN, and MxN <= 256
#
# 16x16 matrices
#
# Links:
#
# - https://juliagpu.gitlab.io/CUDA.jl/api/kernel/#WMMA
# - https://juliagpu.gitlab.io/CUDA.jl/api/kernel/#LLVM-Intrinsics
# - https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/nvidia-ampere-architecture-whitepaper.pdf
# - https://images.nvidia.com/content/volta-architecture/pdf/volta-architecture-whitepaper.pdf
#
# From Volta whitepaper:
# 
#  * Each tensor core operates on a 4x4 matrix and does $$D = A*B + C$$. A*B requires 64 operations
#  * the CUDA API exposes this as a MMA op requiring a 16x16 matrix operated on by a warp
#  * threads can be individually scheduled at the warp level, allowing for concurrency over all threads
#
# GTX 2070 specs:
#
#  * 288 tensor cores, 4 per processor (from Ampere whitepaper)?
#  * 2304 cuda cores, 36 multiprocessors, 64 cores per processor
#  * warp size 32
#  * 1024 threads per multiprocessor/block. Each multiprocessor has a shared memory
#  * 1024 / 32 = 32 warps per block
#
#  threads and cores are not 1 to 1, threads are scheduled onto cores by the scheduler
#
#  while probably not correct the goroutine model is good way to think about this
#
#  1 core = 1 single precision calculation per cycle
#   
# Example:
# 
#   512x512 matrix
#
#   assuming Float32 (4 bytes)
#
#   512*512*4 = 1MB
#
#   each block can have 65536 shared memory
#
#   so we need 16 blocks to perform this computation at full capacity
#
#  Info:
# │   gemm_shape = (M = 4096, N = 4096, K = 4096)
# │   block_shape = (M = 128, N = 128, K = 64)
# │   warps_per_block = 8
# │   mem_a_warp = (M = 128, K = 2)
# │   mem_a_thread = (M = 8, K = 1)
# │   mem_b_warp = (K = 64, N = 4)
# │   mem_b_thread = (K = 8, N = 1)
# │   mem_cd_warp = (M = 128, N = 1)
# │   mem_cd_thread = (M = 4, N = 1)
# │   compute_warp = (M = 32, N = 64, K = 16)
# └   op_shape = (M = 16, N = 16, K = 16)
#
# for this configuration the kernel is launched with 256 threads, since there's
# 8 warps per block, all `*_warp` sizes product = 256
#
# What does this mean ???
#
# - `gemm_shape` is the shape of the matrix dimensions (A, B, C)
# - `block_shape` is the block pieces for each dimension, each block
# - `is operated on in parallel
# - `*_warp` is the parallel workload for each warp. This based off the block_shape. For example: (M = 128, K = 2) means the work is split among (128/128) = 1 piece for the M dimension and (64/2) = 32 pieces for the K dimension. Note we have 8 warps per block so the this shows a bottleneck in processing pieces since we can only do 8 pieces at a time.
# - `*_thread` is the parallel workload for each thread. This based off *_warp. For example: (M = 128, K = 1) means the work is split among (128/8) = 16 pieces of work for the M dimension and (2/1) = 2 pieces for the K dimension. Note we have 32 threads per warp, so all 8x1 pieces can be processed in parallel.


using CUDA, LinearAlgebra, GemmKernels, GemmKernels.Tiling
# using KernelAbstractions
# using KernelAbstractions.Extras: @unroll
# using GPUifyLoops: @unroll
using StaticArrays

CUDA.math_mode!(CUDA.FAST_MATH; precision=:Float16)

M = 4096
N = 4096
K = 4096

# A is MxK
# B is KxN
# C, D are MxN
a_cpu = Float16.(rand(M, K));
a = CuArray{Float16}(Diagonal(a_cpu));
b = CuArray{Float16}(rand(K, N));
c = CuArray{Float32}(zeros(M, N));
d = similar(c);

# bm = Int(M / 128)
# bk = Int(K / 64)
# bs = SMatrix{bm, bk}(CuArray{Float32}(rand(bm , bk) .> 0.99));
# bs = MMatrix{bm, bk}(Array{Float32}(rand(bm , bk) .> 0.99));

conf = GemmKernels.get_config(
    gemm_shape = (M = M, N = N, K = K),
    operator = Operator.WMMAOp{16, 16, 16},
    # global_a_layout = Layout.Diagonal{Float16}(),
    # global_a_layout = Layout.BlockSparse{Float16,0.70}(),
    # global_a_layout = Layout.BlockSparse(bs),

    global_a_layout = Layout.AlignedColMajor{Float16}(),
    global_b_layout = Layout.AlignedColMajor{Float16}(),

    global_c_layout = Layout.AlignedColMajor{Float32}(),
    global_d_layout = Layout.AlignedColMajor{Float32}(),

    shared_a_layout = Layout.Padded{Layout.AlignedColMajor{Float16}, 8}(),

    is_a_col_major = true,
    is_b_col_major = true,
);

struct WMMACall{M,N,K,T<:Union{Float16,Float32}}
end

function mma(::WMMACall{M,N,K,T}, a, b, c, d) where {M, N, K, T}
    wmma_kernel(a, b, c, d, WMMA.Config{M,N,K,T})
end

function wmma_kernel(a, b, c, d, conf)
    # x = threadIdx().x    # this example only requires linear indexing, so just use `x`
    # y = threadIdx().y    # this example only requires linear indexing, so just use `x`
    # stridex = blockIdx().x
    # stridey = blockIdx().y
    # @CUDA.cuprintln("thread $x, $y, block $stridex, $stridey")
    a_frag = WMMA.load_a(pointer(a), 16, WMMA.ColMajor, conf)
    b_frag = WMMA.load_b(pointer(b), 16, WMMA.ColMajor, conf)
    c_frag = WMMA.load_c(pointer(c), 16, WMMA.ColMajor, conf)
    d_frag = WMMA.mma(a_frag, b_frag, c_frag, conf)
    WMMA.store_d(pointer(d), d_frag, 16, WMMA.ColMajor, conf)
    return
end


function divergence1(a, threads_in_block)
    x = threadIdx().x
    bx = blockIdx().x - 1
    idx = bx * threads_in_block + x
    idx <= length(a) || return
    # idx = x
    warp = x ÷ 32
    lane = x % 32

    # @CUDA.cuprintln("thread $x, block $bx, warp $warp, lane $lane")

    if lane <= 4
        a[idx] += 1
    elseif lane <= 16
        a[idx] += 2
    elseif lane <= 23
        a[idx] -= 3
    elseif lane <= 29
        a[idx] += 10
    else
        a[idx] *= 1.2
    end

    return
end

@kernel function i1(a)
    I = @index(Global, Linear)
    @print("index $I\n")
end

@kernel function ka_global_index(a)
    I1 = @index(Global, Linear)
    I2 = @index(Global, Cartesian)
    I3 = @index(Global, NTuple)
    @print("index $I1 $(I2...) $(I3...)\n")
end

@kernel function divergence1(a)
    x = threadIdx().x
    bx = blockIdx().x - 1
    # idx = bx * threads_in_block + x
    idx <= length(a) || return
    # idx = x
    warp = x ÷ 32
    lane = x % 32
    @index(Global)

    # @CUDA.cuprintln("thread $x, block $bx, warp $warp, lane $lane")

    if lane <= 4
        a[idx] += 1
    elseif lane <= 16
        a[idx] += 2
    elseif lane <= 23
        a[idx] -= 3
    elseif lane <= 29
        a[idx] += 10
    else
        a[idx] *= 1.2
    end

    return
end

function divergence2(a, threads_in_block)
    x = threadIdx().x
    bx = blockIdx().x - 1
    idx = bx * threads_in_block + x
    idx <= length(a) || return
    # idx = x
    warp = x ÷ 32
    lane = x % 32

    @CUDA.cuprintln("idx $idx, thread $x, block $bx, warp $warp, lane $lane")

    a[idx] += 1

    return
end


sparsity = 0.99
bm = Int(M / 128)
bk = Int(K / 64)

# bs = SMatrix{bm, bk}(Array{Float32}(rand(bm , bk) .> sparsity))

conf = GemmKernels.get_config(
    gemm_shape = (M = M, N = N, K = K),
    operator = Operator.WMMAOp{16, 16, 16},
    global_a_layout = Layout.BlockSparse{Float32}(bs),
    global_b_layout = Layout.AlignedColMajor{Float16}(),

    global_c_layout = Layout.AlignedColMajor{Float32}(),
    global_d_layout = Layout.AlignedColMajor{Float32}(),

    shared_a_layout = Layout.Padded{Layout.AlignedColMajor{Float16}, 8}(),

    is_a_col_major = true,
    is_b_col_major = true,
)


a_cpu = Float16.(rand(M, K));
a = CuArray{Float16}(Diagonal(a_cpu));
b = CuArray{Float16}(rand(K, N));
c = CuArray{Float32}(zeros(M, N));
d = similar(c);

