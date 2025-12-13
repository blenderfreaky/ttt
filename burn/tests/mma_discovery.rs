//! Test to check whether CubeCL discovers MMA (Matrix Multiply-Accumulate) on the GPU.
//!
//! On RDNA GPUs, MMA should NOT be available (only RDNA3 has WMMA support).

use burn::prelude::*;
use burn::tensor::Tensor;
use burn_cubecl::CubeBackend;
use cubecl::hip::HipRuntime;

type GpuBackend = CubeBackend<HipRuntime, f32, i32, u8>;

#[test]
fn test_mma_discovery() {
    let device: <GpuBackend as Backend>::Device = Default::default();

    // Create a dummy tensor to get access to the client
    let tensor: Tensor<GpuBackend, 1> = Tensor::zeros([1], &device);
    let prim = tensor.into_primitive().tensor();
    let client = &prim.client;

    // Get device properties and features
    let props = client.properties();
    let features = &props.features;

    println!("=== CubeCL MMA Discovery Test ===");
    println!();

    // Print hardware properties
    println!("Hardware properties:");
    println!("  {:?}", props.hardware);
    println!();

    // Check CMMA (Cooperative Matrix Multiply-Accumulate)
    println!("CMMA configurations: {}", features.cmma.len());
    for config in &features.cmma {
        println!("  CMMA: {:?}", config);
    }

    // Check MMA (manual data movement matrix multiply)
    println!("MMA configurations: {}", features.mma.len());
    for config in &features.mma {
        println!("  MMA: {:?}", config);
    }

    // Check Scaled MMA (quantized)
    println!("Scaled MMA configurations: {}", features.scaled_mma.len());
    for config in &features.scaled_mma {
        println!("  Scaled MMA: {:?}", config);
    }

    // Print other features
    println!();
    println!("Other features:");
    println!("  Plane: {:?}", features.plane);

    // Summary
    let has_any_mma =
        !features.cmma.is_empty() || !features.mma.is_empty() || !features.scaled_mma.is_empty();

    println!();
    if has_any_mma {
        println!("✓ MMA support FOUND on this GPU");
        println!("  WARNING: RX 6800 (RDNA2) should NOT have WMMA - this may be a bug in CubeCL!");
    } else {
        println!("✗ No MMA support found on this GPU (expected for RDNA2)");
    }
    println!();
}
