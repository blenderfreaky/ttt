#include <Kokkos_Core.hpp>
#include <iostream>

// Matrix dimensions
constexpr int N = 1024;

int main(int argc, char* argv[]) {
    Kokkos::initialize(argc, argv);
    {
        // Create views for matrices
        Kokkos::View<double**> A("A", N, N);
        Kokkos::View<double**> B("B", N, N);
        Kokkos::View<double**> C("C", N, N);

        // Initialize matrices A and B on host
        auto h_A = Kokkos::create_mirror_view(A);
        auto h_B = Kokkos::create_mirror_view(B);
        
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                h_A(i, j) = 1.0;  // Simple initialization
                h_B(i, j) = 1.0;
            }
        }

        // Copy data to device
        Kokkos::deep_copy(A, h_A);
        Kokkos::deep_copy(B, h_B);

        // Perform matrix multiplication using parallel_for
        Kokkos::parallel_for("matmul", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {N, N}),
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

        // Verify result (all values should be N since A and B are filled with 1.0)
        bool correct = true;
        for (int i = 0; (i < N) && correct; i++) {
            for (int j = 0; (j < N) && correct; j++) {
                if (std::abs(h_C(i, j) - N) > 1e-10) {
                    correct = false;
                    std::cout << "Error at position (" << i << "," << j << "): "
                              << h_C(i, j) << " != " << N << std::endl;
                }
            }
        }

        if (correct) {
            std::cout << "Matrix multiplication completed successfully!" << std::endl;
        }
    }
    Kokkos::finalize();
    return 0;
}