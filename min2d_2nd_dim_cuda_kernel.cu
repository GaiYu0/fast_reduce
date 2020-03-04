#include <cooperative_groups.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <torch/extension.h>

namespace cg = cooperative_groups;

// Utility class used to avoid linker errors with extern
// unsized shared memory arrays with templated type
template <class T>
struct SharedMemory {
  __device__ inline operator T *() {
    extern __shared__ int __smem[];
    return (T *)__smem;
  }

  __device__ inline operator const T *() const {
    extern __shared__ int __smem[];
    return (T *)__smem;
  }
};

// specialize for double to avoid unaligned memory
// access compile errors
template <>
struct SharedMemory<double> {
  __device__ inline operator double *() {
    extern __shared__ double __smem_d[];
    return (double *)__smem_d;
  }

  __device__ inline operator const double *() const {
    extern __shared__ double __smem_d[];
    return (double *)__smem_d;
  }
};

template <class T> inline T max(T lhs, T rhs) {
  return lhs < rhs ? rhs : lhs;
}

template <class T, unsigned int blockSize, bool nIsPow2>
__global__ void reduce6(T *g_idata, T *g_odata, unsigned int n) {
  // Handle to thread block group
  cg::thread_block cta = cg::this_thread_block();
  T *sdata = SharedMemory<T>();

  // perform first level of reduction,
  // reading from global memory, writing to shared memory
  unsigned int tid = threadIdx.x;
  unsigned int j = blockIdx.y;
  unsigned int i = blockIdx.x * blockSize * 2 + threadIdx.x;
  unsigned int gridSize = blockSize * 2 * gridDim.x;

  T mySum = 0;

  // we reduce multiple elements per thread.  The number is determined by the
  // number of active thread blocks (via gridDim).  More blocks will result
  // in a larger gridSize and therefore fewer elements per thread
  while (i < n) {
    // mySum += g_idata[i];
    mySum = max(mySum, g_idata[j * gridDim.y + i]);

    // ensure we don't read out of bounds -- this is optimized away for powerOf2
    // sized arrays
    // if (nIsPow2 || i + blockSize < n) mySum += g_idata[i + blockSize];
    if (nIsPow2 || i + blockSize < n) mySum = max(mySum, g_idata[j * gridDim.y + i + blockSize]);

    i += gridSize;
  }

  // each thread puts its local sum into shared memory
  sdata[tid] = mySum;
  cg::sync(cta);

  // do reduction in shared mem
  if ((blockSize >= 512) && (tid < 256)) {
    // sdata[tid] = mySum = mySum + sdata[tid + 256];
    sdata[tid] = mySum = max(mySum, sdata[tid + 256]);
  }

  cg::sync(cta);

  if ((blockSize >= 256) && (tid < 128)) {
    // sdata[tid] = mySum = mySum + sdata[tid + 128];
    sdata[tid] = mySum = max(mySum, sdata[tid + 128]);
  }

  cg::sync(cta);

  if ((blockSize >= 128) && (tid < 64)) {
    // sdata[tid] = mySum = mySum + sdata[tid + 64];
    sdata[tid] = mySum = max(mySum, sdata[tid + 64]);
  }

  cg::sync(cta);

  cg::thread_block_tile<32> tile32 = cg::tiled_partition<32>(cta);

  if (cta.thread_rank() < 32) {
    // Fetch final intermediate sum from 2nd warp
    // if (blockSize >= 64) mySum += sdata[tid + 32];
    if (blockSize >= 64) mySum = max(mySum, sdata[tid + 32]);
    // Reduce final warp using shuffle
    for (int offset = tile32.size() / 2; offset > 0; offset /= 2) {
      // mySum += tile32.shfl_down(mySum, offset);
      mySum = max(mySum, tile32.shfl_down(mySum, offset));
    }
  }

  // write result for this block to global mem
  if (cta.thread_rank() == 0) g_odata[j * gridDim.y + blockIdx.x] = mySum;
}

torch::Tensor min2d_2nd_dim_cuda_forward(torch::Tensor x) {
  int n = x.size(0);
  int d = x.size(1);
  const int n_threads = 1024;
  int e = d / n_threads + !(d % n_threads);

  dim3 d_block(n_threads, 1, 1);
  dim3 d_grid(n, e, 1);
	int sizeof_dtype;
  switch (tensor.type().scalarType()) {
  case torch::ScalarType::Double:
    sizeof_dtype = function<double>(tensor.data<double>());
		break;
  case torch::ScalarType::Float:
    sizeof_dtype = function<float>(tensor.data<float>());
		break;
	default:
		std::cout << "Unrecognized dtype!" << std::endl;
	}
  int sm_size = (n_threads <= 32) ? 2 * n_threads * sizeof_dtype : N_THREADS * sizeof_dtype;

  torch::Tensor y = torch::empty(x.sizes().vec(), x.options());
	AT_DISPATCH_FLOATING_TYPES(gates.type(), "min2d_2nd_dim_cuda_forward", [&] {
  	reduce6<DTYPE, N_THREADS, false><<<d_grid, d_block, sm_size>>>(x, y, D);
	});
  return y;
}
