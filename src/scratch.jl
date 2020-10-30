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
#
using CUDA, LinearAlgebra, GemmKernels, GemmKernels.Tiling, KernelAbstractions
# using KernelAbstractions.Extras: @unroll
using GPUifyLoops: @unroll

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

conf = GemmKernels.get_config(
    gemm_shape = (M = M, N = N, K = K),
    operator = Operator.WMMAOp{16, 16, 16},
    global_a_layout = Layout.Diagonal{Float16}(),
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

# --------
# BlockSparse
# --------

struct BlockSparse{T} <: Tiling.LayoutBase{T}
    mask::CuArray{T,2}
end

@inline function load(::Type{BlockSparse{T}}, workspace, tile::Tile{size}) where {T, size}
    N = 16 ÷ sizeof(T)

    linear_base = linearise(tile.base, Base.size(workspace))
    linear_offset = linearise(tile.offset, Base.size(workspace))

    return vloada(Vec{N, T}, pointer(workspace), linear_base + linear_offset - 1)
end

@inline threadblock_condition(layout_a::Type{BlockSparse{T}}, layout_b, block_i, block_j, block_k, block_tile) where {T} = layout_a[ block_i / block_tile.size.M , block_k / block_tile.size.K] == one(T)


sparsity = 0.99
bs = CuArray{Float16}(rand(Int(M / 128), Int(K / 64)) .> sparsity)

conf = GemmKernels.get_config(
    gemm_shape = (M = M, N = N, K = K),
    operator = Operator.WMMAOp{16, 16, 16},
    global_a_layout = Layout.BlockSparse{Float16}(bs),
    global_b_layout = Layout.AlignedColMajor{Float16},

    global_c_layout = Layout.AlignedColMajor{Float32},
    global_d_layout = Layout.AlignedColMajor{Float32},

    shared_a_layout = Layout.Padded{Layout.AlignedColMajor{Float16}, 8},

    is_a_col_major = true,
    is_b_col_major = true,
)


a_cpu = Float16.(rand(M, K));
a = CuArray{Float16}(Diagonal(a_cpu));
b = CuArray{Float16}(rand(K, N));
c = CuArray{Float32}(zeros(M, N));
d = similar(c);

