#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>
#include <iostream>
#include <sstream>
#include <string>
#include <chrono>

// Force HIP execution space
#ifdef KOKKOS_ENABLE_HIP
using ExecSpace = Kokkos::HIP;
#else
using ExecSpace = Kokkos::DefaultExecutionSpace;
#endif

// Matrix dimensions - make it bigger to better utilize GPU
constexpr int N = 4096;

int main(int argc, char* argv[]) {
    Kokkos::initialize(argc, argv);
    {
        std::cout << "Execution space: " << typeid(Kokkos::DefaultExecutionSpace).name() << std::endl;
        
        // Get available devices
        std::ostringstream msg;
        msg << "Using Kokkos configurations:\n";
#ifdef KOKKOS_ENABLE_HIP
        msg << "HIP enabled\n";
#endif
#ifdef KOKKOS_ENABLE_OPENMP
        msg << "OpenMP enabled\n";
#endif
        std::cout << msg.str();

        // Initialize random number generator
        Kokkos::Random_XorShift64_Pool<> random_pool(12345);

        // Create views for matrices
        Kokkos::View<double**> A("A", N, N);
        Kokkos::View<double**> B("B", N, N);
        Kokkos::View<double**> C("C", N, N);

        // Initialize matrices A and B on host
        auto h_A = Kokkos::create_mirror_view(A);
        auto h_B = Kokkos::create_mirror_view(B);
        
        // Initialize matrices with random values using parallel_for
        Kokkos::parallel_for("init_matrices", Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {N, N}),
            KOKKOS_LAMBDA (const int i, const int j) {
                auto gen = random_pool.get_state();
                A(i, j) = gen.drand(0.0, 1.0);
                B(i, j) = gen.drand(0.0, 1.0);
                random_pool.free_state(gen);
            });

        // Ensure all initialization is complete
        Kokkos::fence();

        // Start timing
        auto start = std::chrono::high_resolution_clock::now();

        // Perform matrix multiplication using parallel_for
        Kokkos::parallel_for("matmul", Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {N, N}),
            KOKKOS_LAMBDA(const int i, const int j) {
                double tmp = 0.0;
                for (int k = 0; k < N; k++) {
                    tmp += A(i, k) * B(k, j);
                }
                C(i, j) = tmp;
            });

        // Copy result back to host
        auto h_C = Kokkos::create_mirror_view(C);
        Kokkos::deep_copy(h_C, C);

        // End timing
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;

        // Verify result by computing the Frobenius norm of a few elements
        double max_diff = 0.0;
        for (int i = 0; i < std::min(10, N); i++) {
            for (int j = 0; j < std::min(10, N); j++) {
                double expected = 0.0;
                for (int k = 0; k < N; k++) {
                    expected += h_A(i, k) * h_B(k, j);
                }
                max_diff = std::max(max_diff, std::abs(h_C(i, j) - expected));
            }
        }

        std::cout << "Matrix multiplication completed!" << std::endl;
        std::cout << "Matrix size: " << N << "x" << N << std::endl;
        std::cout << "Max difference in sampled elements: " << max_diff << std::endl;
        std::cout << "Time: " << elapsed.count() << " seconds" << std::endl;
        double total_ops = 2.0 * N * N * N;  // multiply-add per inner loop iteration
        std::cout << "Total operations: " << total_ops << std::endl;
        std::cout << "Performance: " << (total_ops / elapsed.count() / 1e9) << " GFLOPS" << std::endl;
    }
    Kokkos::finalize();
    return 0;
}