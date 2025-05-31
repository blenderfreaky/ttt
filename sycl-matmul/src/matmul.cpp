#include <sycl/sycl.hpp>
#include <iostream>
#include <vector>
#include <chrono>

// Matrix multiplication kernel
void matrix_multiplication(sycl::queue& q, const std::vector<float>& a,
                         const std::vector<float>& b, std::vector<float>& c,
                         size_t n) {
    sycl::buffer<float, 1> a_buf(a.data(), a.size());
    sycl::buffer<float, 1> b_buf(b.data(), b.size());
    sycl::buffer<float, 1> c_buf(c.data(), c.size());

    q.submit([&](sycl::handler& h) {
        auto a_acc = a_buf.get_access<sycl::access::mode::read>(h);
        auto b_acc = b_buf.get_access<sycl::access::mode::read>(h);
        auto c_acc = c_buf.get_access<sycl::access::mode::write>(h);

        h.parallel_for(sycl::range<2>{n, n}, [=](sycl::id<2> idx) {
            const size_t row = idx[0];
            const size_t col = idx[1];
            float sum = 0.0f;

            for (size_t k = 0; k < n; k++) {
                sum += a_acc[row * n + k] * b_acc[k * n + col];
            }

            c_acc[row * n + col] = sum;
        });
    }).wait();
}

int main() {
    const size_t n = 1024; // Matrix size (n x n)
    std::vector<float> a(n * n), b(n * n), c(n * n);

    // Initialize matrices
    for (size_t i = 0; i < n * n; i++) {
        a[i] = static_cast<float>(i % 7);
        b[i] = static_cast<float>(i % 5);
        c[i] = 0.0f;
    }

    try {
        // Try to find any available device
        auto devices = sycl::device::get_devices();
        if (devices.empty()) {
            throw std::runtime_error("No SYCL devices found");
        }

        // Print available devices
        std::cout << "Available devices:\n";
        for (const auto& dev : devices) {
            std::cout << " - " << dev.get_info<sycl::info::device::name>() << "\n";
        }

        // Create queue with the first available device
        sycl::queue q{devices[0]};
        std::cout << "\nUsing device: "
                 << q.get_device().get_info<sycl::info::device::name>() << "\n";

        // Perform matrix multiplication with timing
        auto start = std::chrono::high_resolution_clock::now();
        
        matrix_multiplication(q, a, b, c, n);
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

        std::cout << "Matrix multiplication completed in " << duration.count() 
                 << " ms\n";

        // Verify result (check a few elements)
        std::cout << "\nSample results (top-left 3x3):\n";
        for (size_t i = 0; i < 3; i++) {
            for (size_t j = 0; j < 3; j++) {
                std::cout << c[i * n + j] << " ";
            }
            std::cout << "\n";
        }

        // Verify result with CPU computation for the first few elements
        std::cout << "\nVerifying results...\n";
        bool verification_passed = true;
        for (size_t i = 0; i < 3 && verification_passed; i++) {
            for (size_t j = 0; j < 3 && verification_passed; j++) {
                float expected = 0.0f;
                for (size_t k = 0; k < n; k++) {
                    expected += a[i * n + k] * b[k * n + j];
                }
                if (std::abs(c[i * n + j] - expected) > 1e-5) {
                    verification_passed = false;
                    std::cout << "Verification failed at position (" << i << "," << j 
                              << "): expected " << expected << ", got " << c[i * n + j] << "\n";
                }
            }
        }
        
        if (verification_passed) {
            std::cout << "Result verification passed!\n";
        }

    } catch (const sycl::exception& e) {
        std::cerr << "SYCL exception caught: " << e.what() << "\n";
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "Standard exception caught: " << e.what() << "\n";
        return 2;
    }

    return 0;
}