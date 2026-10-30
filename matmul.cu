// Learning order: CPU basic -> reordered -> blocked -> CUDA naive -> shared
// memory -> register tiling -> cuBLAS. All compute row-major C = A * B in FP32.
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

void cuda_check(cudaError_t status, const char* call) {
    if (status != cudaSuccess)
        throw std::runtime_error(std::string(call) + ": " + cudaGetErrorString(status));
}
void blas_check(cublasStatus_t status, const char* call) {
    if (status != CUBLAS_STATUS_SUCCESS)
        throw std::runtime_error(std::string(call) + ": cuBLAS status " + std::to_string(status));
}
#define CUDA(call) cuda_check((call), #call)
#define BLAS(call) blas_check((call), #call)
using Matrix = std::vector<float>;

// STAGE 1: The mathematical definition. B is read down a column (stride n).
void cpu_basic(const Matrix& a, const Matrix& b, Matrix& c, int n) {
    for (int row = 0; row < n; ++row)
        for (int col = 0; col < n; ++col) {
            float sum = 0;
            for (int k = 0; k < n; ++k)
                sum += a[size_t(row) * n + k] * b[size_t(k) * n + col];
            c[size_t(row) * n + col] = sum;
        }
}

// STAGE 2: i-k-j loop order streams across rows of B and C, enabling SIMD.
void cpu_reordered(const Matrix& a, const Matrix& b, Matrix& c, int n) {
    std::fill(c.begin(), c.end(), 0.0f);
    for (int row = 0; row < n; ++row)
        for (int k = 0; k < n; ++k) {
            const float av = a[size_t(row) * n + k];
            for (int col = 0; col < n; ++col)
                c[size_t(row) * n + col] += av * b[size_t(k) * n + col];
        }
}

// STAGE 3: Keep small submatrices in the CPU caches. Still single threaded.
void cpu_blocked(const Matrix& a, const Matrix& b, Matrix& c, int n) {
    constexpr int tile = 32;
    std::fill(c.begin(), c.end(), 0.0f);
    for (int ii = 0; ii < n; ii += tile)
        for (int kk = 0; kk < n; kk += tile)
            for (int jj = 0; jj < n; jj += tile)
                for (int i = ii; i < std::min(ii + tile, n); ++i)
                    for (int k = kk; k < std::min(kk + tile, n); ++k) {
                        const float av = a[size_t(i) * n + k];
                        for (int j = jj; j < std::min(jj + tile, n); ++j)
                            c[size_t(i) * n + j] += av * b[size_t(k) * n + j];
                    }
}

// STAGE 4: One thread per output. x maps to columns, y maps to rows.
__global__ void gpu_naive(const float* __restrict__ a,
                          const float* __restrict__ b, float* c, int n) {
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n || col >= n) return;
    float sum = 0;
    for (int k = 0; k < n; ++k)
        sum = fmaf(a[size_t(row) * n + k], b[size_t(k) * n + col], sum);
    c[size_t(row) * n + col] = sum;
}

// STAGE 5: Each block loads two tiles, then reuses each value T times.
template<int T>
__global__ void gpu_shared(const float* __restrict__ a,
                           const float* __restrict__ b, float* c, int n) {
    __shared__ float as[T][T];
    __shared__ float bs[T][T];
    const int x = threadIdx.x, y = threadIdx.y;
    const int row = blockIdx.y * T + y, col = blockIdx.x * T + x;
    float sum = 0;
    for (int base = 0; base < n; base += T) {
        as[y][x] = (row < n && base + x < n) ? a[size_t(row) * n + base + x] : 0;
        bs[y][x] = (base + y < n && col < n) ? b[size_t(base + y) * n + col] : 0;
        __syncthreads(); // All threads, including out-of-range output threads.
        #pragma unroll
        for (int k = 0; k < T; ++k) sum = fmaf(as[y][k], bs[k][x], sum);
        __syncthreads(); // Finish reading before anyone overwrites the tiles.
    }
    if (row < n && col < n) c[size_t(row) * n + col] = sum;
}

// STAGE 6: 256 threads compute a 64x64 output tile using a 4x4 register
// tile per thread. A/B K tiles are 64x16 / 16x64. Cooperative linear loads
// access consecutive global addresses; each warp spans two 16-thread rows.
// Accumulator columns are spaced by 16 so neighboring x lanes stay adjacent.
__global__ void gpu_register(const float* __restrict__ a,
                             const float* __restrict__ b, float* c, int n) {
    constexpr int BM = 64, BN = 64, BK = 16, TM = 4, TN = 4;
    __shared__ float as[BM][BK];
    __shared__ float bs[BK][BN];
    const int tx = threadIdx.x, ty = threadIdx.y;
    const int tid = ty * 16 + tx;
    const int row0 = blockIdx.y * BM, col0 = blockIdx.x * BN;
    float acc[TM][TN] = {};
    for (int base = 0; base < n; base += BK) {
        for (int idx = tid; idx < BM * BK; idx += 256) {
            const int r = idx / BK, k = idx % BK;
            as[r][k] = (row0 + r < n && base + k < n)
                ? a[size_t(row0 + r) * n + base + k] : 0;
        }
        for (int idx = tid; idx < BK * BN; idx += 256) {
            const int k = idx / BN, col = idx % BN;
            bs[k][col] = (base + k < n && col0 + col < n)
                ? b[size_t(base + k) * n + col0 + col] : 0;
        }
        __syncthreads();
        #pragma unroll
        for (int k = 0; k < BK; ++k) {
            float ar[TM], br[TN];
            #pragma unroll
            for (int i = 0; i < TM; ++i) ar[i] = as[ty + i * 16][k];
            #pragma unroll
            for (int j = 0; j < TN; ++j) br[j] = bs[k][tx + j * 16];
            #pragma unroll
            for (int i = 0; i < TM; ++i)
                #pragma unroll
                for (int j = 0; j < TN; ++j)
                    acc[i][j] = fmaf(ar[i], br[j], acc[i][j]);
        }
        __syncthreads();
    }
    #pragma unroll
    for (int i = 0; i < TM; ++i)
        #pragma unroll
        for (int j = 0; j < TN; ++j) {
            const int row = row0 + ty + i * 16, col = col0 + tx + j * 16;
            if (row < n && col < n) c[size_t(row) * n + col] = acc[i][j];
        }
}

struct DeviceBuffer {
    float* ptr = nullptr;
    explicit DeviceBuffer(size_t bytes) { CUDA(cudaMalloc(&ptr, bytes)); }
    ~DeviceBuffer() { if (ptr) cudaFree(ptr); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
};
struct Event {
    cudaEvent_t value{};
    Event() { CUDA(cudaEventCreate(&value)); }
    ~Event() { cudaEventDestroy(value); }
    Event(const Event&) = delete;
    Event& operator=(const Event&) = delete;
};
struct BlasHandle {
    cublasHandle_t value{};
    BlasHandle() {
        BLAS(cublasCreate(&value));
        // Keep FP32 arithmetic comparable; don't silently enable TF32.
        const auto status = cublasSetMathMode(value, CUBLAS_PEDANTIC_MATH);
        if (status != CUBLAS_STATUS_SUCCESS) {
            cublasDestroy(value);
            blas_check(status, "cublasSetMathMode");
        }
    }
    ~BlasHandle() { cublasDestroy(value); }
};

// More accurate oracle for the small correctness suite, independent of CPU FP32.
Matrix reference_double(const Matrix& a, const Matrix& b, int n) {
    Matrix c(size_t(n) * n);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j) {
            double sum = 0;
            for (int k = 0; k < n; ++k)
                sum += double(a[size_t(i) * n + k]) * double(b[size_t(k) * n + j]);
            c[size_t(i) * n + j] = float(sum);
        }
    return c;
}
void validate(const Matrix& got, const Matrix& expected, const std::string& name) {
    float max_error = 0;
    for (size_t i = 0; i < got.size(); ++i) {
        const float error = std::abs(got[i] - expected[i]);
        max_error = std::max(error, max_error);
        if (!std::isfinite(got[i]) || error > 1e-3f + 1e-3f * std::abs(expected[i]))
            throw std::runtime_error(name + " mismatch at index " + std::to_string(i)
                + ": got " + std::to_string(got[i]) + ", expected " + std::to_string(expected[i]));
    }
    std::cout << "  " << name << " PASS (max abs error " << max_error << ")\n";
}
void fill_inputs(Matrix& a, Matrix& b, int pattern) {
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    const int n = int(std::sqrt(double(a.size())));
    for (size_t i = 0; i < a.size(); ++i) {
        a[i] = pattern == 1 ? ((i / n == i % n) ? 1.0f : 0.0f) : dist(rng);
        b[i] = pattern == 2 ? 0.0f : dist(rng);
    }
}
struct Variant {
    std::string name;
    std::function<void()> launch;
};
std::vector<Variant> gpu_variants(float* a, float* b, float* c, int n, cublasHandle_t handle) {
    auto grid = [n](int t) { return dim3((n + t - 1) / t, (n + t - 1) / t); };
    return {
        {"gpu_naive", [=] { gpu_naive<<<grid(16), dim3(16,16)>>>(a,b,c,n); }},
        {"gpu_shared8", [=] { gpu_shared<8><<<grid(8), dim3(8,8)>>>(a,b,c,n); }},
        {"gpu_shared16", [=] { gpu_shared<16><<<grid(16), dim3(16,16)>>>(a,b,c,n); }},
        {"gpu_shared32", [=] { gpu_shared<32><<<grid(32), dim3(32,32)>>>(a,b,c,n); }},
        {"gpu_register", [=] { gpu_register<<<grid(64), dim3(16,16)>>>(a,b,c,n); }},
        {"cublas", [=] {
            const float alpha = 1, beta = 0;
            // cuBLAS sees column-major matrices: C^T = B^T * A^T.
            BLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                            n,n,n, &alpha, b,n, a,n, &beta, c,n));
        }}
    };
}

template<typename Kernel>
void print_occupancy(const char* name, Kernel kernel, int threads, const cudaDeviceProp& prop) {
    int blocks = 0;
    cudaFuncAttributes attr{};
    CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, kernel, threads, 0));
    CUDA(cudaFuncGetAttributes(&attr, kernel));
    const double occupancy = 100.0 * blocks * (threads / prop.warpSize)
        / (prop.maxThreadsPerMultiProcessor / prop.warpSize);
    std::cout << name << ": threads/block=" << threads << ", registers/thread=" << attr.numRegs
              << ", shared bytes/block=" << attr.sharedSizeBytes
              << ", resident blocks/SM=" << blocks << ", theoretical occupancy=" << occupancy << "%\n";
}

void run(int n, int iterations, const std::string& only, bool test, int pattern, bool cpu_only) {
    const size_t count = size_t(n) * n, bytes = count * sizeof(float);
    Matrix a(count), b(count), c(count), expected(count);
    fill_inputs(a,b,pattern);
    std::cout << "\nN=" << n << ", iterations=" << iterations << ", pattern=" << pattern << '\n';
    if (test) expected = reference_double(a,b,n);
    else {
        auto begin = std::chrono::steady_clock::now();
        cpu_basic(a,b,expected,n);
        double ms = std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begin).count();
        std::cout << "CPU FP32 reference: " << ms << " ms (separate validation pass)\n";
    }
    auto report = [=](const std::string& name, double ms) {
        std::cout << std::fixed << std::setprecision(4) << "RESULT " << name
                  << " ms=" << ms << " GFLOP/s=" << (2.0 * n * n * n / (ms * 1e6)) << '\n';
    };
    const std::vector<std::pair<std::string, std::function<void()>>> cpus = {
        {"cpu_basic", [&] { cpu_basic(a,b,c,n); }},
        {"cpu_reordered", [&] { cpu_reordered(a,b,c,n); }},
        {"cpu_blocked", [&] { cpu_blocked(a,b,c,n); }}
    };
    for (const auto& v : cpus) {
        if (only != "all" && only != v.first) continue;
        v.second(); // warm-up
        const auto start = std::chrono::steady_clock::now();
        for (int rep = 0; rep < iterations; ++rep) v.second();
        const double ms = std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count()/iterations;
        validate(c, expected, v.first);
        report(v.first, ms);
    }
    if (cpu_only || only.rfind("cpu_", 0) == 0) return;
    DeviceBuffer da(bytes), db(bytes), dc(bytes);
    CUDA(cudaMemcpy(da.ptr,a.data(),bytes,cudaMemcpyHostToDevice));
    CUDA(cudaMemcpy(db.ptr,b.data(),bytes,cudaMemcpyHostToDevice));
    BlasHandle handle;
    for (const auto& v : gpu_variants(da.ptr,db.ptr,dc.ptr,n,handle.value)) {
        if (only != "all" && only != v.name) continue;
        CUDA(cudaMemset(dc.ptr, 0xff, bytes)); // NaNs expose missed output stores.
        v.launch();
        CUDA(cudaGetLastError());
        CUDA(cudaDeviceSynchronize());
        CUDA(cudaMemcpy(c.data(),dc.ptr,bytes,cudaMemcpyDeviceToHost));
        validate(c,expected,v.name);
        Event start, stop;
        CUDA(cudaEventRecord(start.value));
        for (int rep = 0; rep < iterations; ++rep) v.launch();
        CUDA(cudaGetLastError());
        CUDA(cudaEventRecord(stop.value));
        CUDA(cudaEventSynchronize(stop.value));
        float total_ms = 0;
        CUDA(cudaEventElapsedTime(&total_ms,start.value,stop.value));
        report(v.name,total_ms/iterations);
    }
}

int positive_int(const char* text) {
    size_t consumed = 0;
    const std::string s(text);
    const int value = std::stoi(s,&consumed);
    if (consumed != s.size() || value <= 0) throw std::runtime_error("Expected a positive integer: " + s);
    return value;
}
int main(int argc, char** argv) {
    try {
        int n = 256, iterations = 5;
        bool test = false, cpu_only = false;
        std::string only = "all";
        for (int i = 1; i < argc; ++i) {
            const std::string arg(argv[i]);
            if (arg == "--test") test = true;
            else if (arg == "--cpu-only") cpu_only = true;
            else if (arg == "--n" && i+1 < argc) n = positive_int(argv[++i]);
            else if (arg == "--iterations" && i+1 < argc) iterations = positive_int(argv[++i]);
            else if (arg == "--only" && i+1 < argc) only = argv[++i];
            else if (arg == "--help") {
                std::cout << "matmul [--n N] [--iterations R] [--only STAGE] [--test] [--cpu-only]\n"
                          << "STAGE: all, cpu_basic, cpu_reordered, cpu_blocked, gpu_naive,\n"
                          << "       gpu_shared8, gpu_shared16, gpu_shared32, gpu_register, cublas\n"
                          << "--test runs all stages on boundary sizes and input patterns.\n";
                return 0;
            } else throw std::runtime_error("Unknown or incomplete argument: " + arg);
        }
        const std::vector<std::string> names = {"all", "cpu_basic", "cpu_reordered", "cpu_blocked",
            "gpu_naive", "gpu_shared8", "gpu_shared16", "gpu_shared32", "gpu_register", "cublas"};
        if (n > 16384) throw std::runtime_error("N must be <= 16384 (demo memory/indexing limit)");
        if (std::find(names.begin(),names.end(),only) == names.end()) throw std::runtime_error("Unknown stage: " + only);
        if (cpu_only && only != "all" && only.rfind("cpu_",0) != 0)
            throw std::runtime_error("--cpu-only requires a CPU stage or all");
        if (test && only != "all") throw std::runtime_error("--test runs all stages; omit --only");
        if (!cpu_only && only.rfind("cpu_",0) != 0) {
            cudaDeviceProp prop{};
            CUDA(cudaGetDeviceProperties(&prop,0));
            CUDA(cudaSetDevice(0));
            std::cout << "GPU: " << prop.name << " (compute " << prop.major << '.' << prop.minor << ")\n";
            print_occupancy("gpu_naive",gpu_naive,256,prop);
            print_occupancy("gpu_shared8",gpu_shared<8>,64,prop);
            print_occupancy("gpu_shared16",gpu_shared<16>,256,prop);
            print_occupancy("gpu_shared32",gpu_shared<32>,1024,prop);
            print_occupancy("gpu_register",gpu_register,256,prop);
        }
        if (test) {
            for (int size : {1, 7, 16, 17, 31, 32, 33, 63, 64, 65, 129, 513})
                run(size,1,"all",true,0,cpu_only);
            for (int pattern : {1,2}) run(33,1,"all",true,pattern,cpu_only);
            std::cout << "\nALL CORRECTNESS CHECKS PASSED\n";
        } else run(n,iterations,only,false,0,cpu_only);
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << '\n';
        return 1;
    }
}
