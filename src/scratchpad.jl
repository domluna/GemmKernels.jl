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
# Tensor Core:
# 
#  * Each tensor core operates on a 4x4 matrix and does $$D = A*B + C$$ **per clock cycle**.
#  * the CUDA API exposes this as a MMA op requiring a 16x16 matrix operated on by a warp
#
# Threads:
#  * threads can be individually scheduled at the warp level, allowing for concurrency over all threads note
#
# GTX 2070 specs:
#
#  * 2304 cuda cores, 36 multiprocessors, 64 cores per multiprocessor
#  * 288 tensor cores, 8 per multiprocessor
#  * 32 threads per warp
#  * 1024 threads per multiprocessor/block. Each multiprocessor has a shared memory
#  * 1024 / 32 = 32 warps per block
#  * 1 core = 1 single precision calculation per cycle
#
#  Threads and cores are not 1 to 1, threads are scheduled onto cores by the scheduler.
#  While not correct, the task concurrency model is a good way to think about this.
#  You can have many more tasks than cores active at a time, but only at most N number
#  of tasks have work being done on them where N is the number of cores.
#
#   
# Example:
# 
#   4096x4096 matrix
#
#   assuming Float32 (4 bytes)
#
#   4096*4096*4 = 64MB
#
#   each block can have 65536 bytes of shared memory
#
#   so we need 1024 blocks to perform this computation at full capacity
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
# - `op_shape` reflects that the WMMA tensor core op API is for 16x16 matrices
# - `compute_warp` is the parallel tensor core workload for each warp. Since WMMA uses 16x16 matrices for a warp this means each warp will perform: 32/16 = 2 pieces along M, 64/16 = 4 pieces along N, 16/16 = 1 pieces along K. This means there's going to be a bottleneck over the tensor cores since only 2 calls can be processed in parallel.
#
# Since each block has 8 tensor cores only 2 warps can operate in parallel. This is because each warp uses 4 tensors cores for a 16x16 matrix.


# using KernelAbstractions
# using KernelAbstractions.Extras: @unroll
# using GPUifyLoops: @unroll
# using StaticArrays
using CUDA, LinearAlgebra, GemmKernels, GemmKernels.Tiling

CUDA.math_mode!(CUDA.FAST_MATH; precision=:Float16)

M, N, K = 4096, 4096, 4096
# M, N, K = 8192, 8192, 8192
# M, N, K = 1024, 1024, 1024

# A is MxK
# B is KxN
# C, D are MxN
a = CuArray{Float16}(rand(M, K));
b = CuArray{Float16}(rand(K, N));
c = CuArray{Float32}(zeros(M, N));
d = similar(c);

conf = GemmKernels.get_config(
    gemm_shape = (M = M, N = N, K = K),
    operator = Operator.WMMAOp{16, 16, 16},

    global_a_layout = Layout.AlignedColMajor{eltype(a)}(),
    global_b_layout = Layout.AlignedColMajor{eltype(b)}(),

    global_c_layout = Layout.AlignedColMajor{eltype(c)}(),
    global_d_layout = Layout.AlignedColMajor{eltype(d)}(),

    is_a_col_major = true,
    is_b_col_major = true,

    gemm_shape = (M = 32, N = 32, K = 16),
    warps_per_blocks=16,
);

bm = CuArray{Float32}(ones(32,32));
out = CuArray{Float32}(zeros(size(bm)));

function blocksparse(bm, out)
    block_m = blockIdx().x
    block_n = blockIdx().y
    t = Tile((M=1, N=1))

    val = Layout.load(Layout.AlignedColMajor{Float32}, bm, translate_base(t, (M=block_m-1, N = block_n-1)))
    # val = Layout.load(Layout.Padded{Layout.AlignedColMajor{Float32}, 8}, bm, translate_base(t, (M=block_m, N = block_n)))

    # @CUDA.cuprintln("block $block_m, $block_n: $(val[1].value), $(val[2].value), $(val[3].value), $(val[4].value)")
    @CUDA.cuprintln("block $block_m, $block_n: $(val[1].value)")

    # this isn't value the right value unless it's NTuple{1,VecElement{Float32}}
    if val[1].value == 1
        out[block_m, block_n] = 55.2
    end

    return nothing
end

struct WMMACall{M,N,K,T<:Union{Float16,Float32}}
end

function mma(::WMMACall{M,N,K,T}, a, b, c, d) where {M, N, K, T}
    wmma_kernel(a, b, c, d, WMMA.Config{M,N,K,T})
end

function wmma_kernel(a, b, d)
    conf = WMMA.Config{16,16,16,Float32}
    a_frag = WMMA.load_a(pointer(a), 16, WMMA.ColMajor, conf)
    b_frag = WMMA.load_b(pointer(b), 16, WMMA.ColMajor, conf)
    c_frag = WMMA.fill_c(Float32(0), conf)
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


# offset: (M = 0, N = 0, K = 0)
# size:   (M = 128, N = 128, K = 64))), $(QuoteNode(base:   (M = 0, K = 0, N = 0)
# offset: (M = 0, K = 0, N = 0)
# size:   (M = 32, K = 16, N = 32))), %31, $(Expr(:static_parameter, 3)))
