#include <sycl/sycl.hpp>
#include <iostream>
#include <vector>
#include <chrono>
#include <algorithm>
#include <numeric>
#include <iomanip>

// Configuration constants
constexpr size_t TILE_SIZE = 16;
constexpr size_t DEFAULT_MATRIX_SIZE = 1024;
constexpr size_t WARMUP_RUNS = 2;
constexpr size_t BENCHMARK_RUNS = 5;

// Helper function to round up to next multiple
inline size_t round_up(size_t x, size_t y) {
    return ((x + y - 1) / y) * y;
}

// Helper function for timing
template<typename F>
double time_function(F&& func) {
    auto start = std::chrono::high_resolution_clock::now();
    func();
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(end - start).count();
}

// Optimized matrix multiplication using USM and tiling
void matrix_multiplication_optimized(sycl::queue& q, const float* a, const float* b, float* c,
                                   size_t n, size_t work_group_size) {
    const size_t aligned_n = round_up(n, TILE_SIZE);
    const size_t num_tiles = aligned_n / TILE_SIZE;
    
    sycl::range<2> global_range{aligned_n, aligned_n};
    sycl::range<2> local_range{work_group_size, work_group_size};
    
    q.submit([&](sycl::handler& h) {
        h.parallel_for(
            sycl::nd_range<2>{global_range, local_range},
            [=](sycl::nd_item<2> item) {
                const size_t global_x = item.get_global_id(0);
                const size_t global_y = item.get_global_id(1);
                
                if (global_x >= n || global_y >= n) return;
                
                float sum = 0.0f;
                
                // Process the matrix in tiles
                for (size_t t = 0; t < num_tiles; ++t) {
                    // Calculate indices for current tile
                    const size_t tile_start = t * TILE_SIZE;
                    
                    // Process current tile
                    #pragma unroll 4
                    for (size_t k = 0; k < TILE_SIZE; ++k) {
                        const size_t k_idx = tile_start + k;
                        if (k_idx < n) {
                            sum += a[global_x * n + k_idx] * b[k_idx * n + global_y];
                        }
                    }
                }
                
                c[global_x * n + global_y] = sum;
            });
    }).wait();
}

// CPU verification implementation
void matrix_multiplication_cpu(const float* a, const float* b, float* c, size_t n) {
    #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < n; j++) {
            float sum = 0.0f;
            for (size_t k = 0; k < n; k++) {
                sum += a[i * n + k] * b[k * n + j];
            }
            c[i * n + j] = sum;
        }
    }
}

// Verification helper
bool verify_result(const float* a, const float* b, const float* c, size_t n) {
    std::vector<float> c_ref(n * n);
    matrix_multiplication_cpu(a, b, c_ref.data(), n);
    
    const float epsilon = 1e-4f;
    for (size_t i = 0; i < n * n; ++i) {
        if (std::abs(c[i] - c_ref[i]) > epsilon) {
            std::cout << "Verification failed at index " << i 
                     << ": expected " << c_ref[i] 
                     << ", got " << c[i] << "\n";
            return false;
        }
    }
    return true;
}

int main(int argc, char* argv[]) {
    size_t n = DEFAULT_MATRIX_SIZE;
    if (argc > 1) {
        n = std::stoul(argv[1]);
    }
    
    try {
        auto devices = sycl::device::get_devices();
        if (devices.empty()) {
            throw std::runtime_error("No SYCL devices found");
        }
        
        std::cout << "Available devices:\n";
        for (const auto& dev : devices) {
            std::cout << " - " << dev.get_info<sycl::info::device::name>() << "\n";
        }
        
        sycl::property_list props{sycl::property::queue::in_order()};
        sycl::queue q(devices[0], props);
        std::cout << "\nUsing device: "
                  << q.get_device().get_info<sycl::info::device::name>() << "\n";
        
        float* a = sycl::malloc_device<float>(n * n, q);
        float* b = sycl::malloc_device<float>(n * n, q);
        float* c = sycl::malloc_device<float>(n * n, q);
        
        std::vector<float> h_a(n * n), h_b(n * n), h_c(n * n);
        std::iota(h_a.begin(), h_a.end(), 1.0f);
        std::iota(h_b.begin(), h_b.end(), 0.5f);
        
        q.memcpy(a, h_a.data(), n * n * sizeof(float));
        q.memcpy(b, h_b.data(), n * n * sizeof(float));
        q.wait();
        
        const std::vector<size_t> work_group_sizes = {4, 8, 16};
        
        std::cout << "\nMatrix size: " << n << "x" << n << "\n";
        std::cout << "Tile size: " << TILE_SIZE << "x" << TILE_SIZE << "\n\n";
        
        std::cout << std::setw(15) << "Work Group" 
                  << std::setw(15) << "Min Time" 
                  << std::setw(15) << "Max Time"
                  << std::setw(15) << "Avg Time" << "\n";
        std::cout << std::string(60, '-') << "\n";
        
        for (size_t wgs : work_group_sizes) {
            if (wgs > TILE_SIZE) continue;
            
            std::vector<double> timings;
            
            // Warmup runs
            for (size_t i = 0; i < WARMUP_RUNS; ++i) {
                matrix_multiplication_optimized(q, a, b, c, n, wgs);
            }
            
            // Benchmark runs
            for (size_t i = 0; i < BENCHMARK_RUNS; ++i) {
                double time = time_function([&]() {
                    matrix_multiplication_optimized(q, a, b, c, n, wgs);
                });
                timings.push_back(time);
            }
            
            double min_time = *std::min_element(timings.begin(), timings.end());
            double max_time = *std::max_element(timings.begin(), timings.end());
            double avg_time = std::accumulate(timings.begin(), timings.end(), 0.0) / timings.size();
            
            std::cout << std::setw(12) << wgs << "x" << std::left << std::setw(2) << wgs 
                      << std::right << std::setw(15) << std::fixed << std::setprecision(2) << min_time
                      << std::setw(15) << max_time
                      << std::setw(15) << avg_time << "\n";
            
            // Verify result for the last run
            q.memcpy(h_c.data(), c, n * n * sizeof(float)).wait();
            if (!verify_result(h_a.data(), h_b.data(), h_c.data(), n)) {
                std::cout << "Verification failed for work group size " << wgs << "\n";
            }
        }
        
        sycl::free(a, q);
        sycl::free(b, q);
        sycl::free(c, q);
        
    } catch (const sycl::exception& e) {
        std::cerr << "SYCL exception caught: " << e.what() << "\n";
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "Standard exception caught: " << e.what() << "\n";
        return 2;
    }
    
    return 0;
}