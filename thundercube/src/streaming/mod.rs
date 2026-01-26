//! Async I/O primitives for persistent kernels.
//!
//! When a kernel is running on the default HIP stream, normal `hipMemcpy` calls
//! block waiting for it to finish. This module provides [`AsyncStream`], which
//! creates a separate non-blocking stream for memory transfers.
//!
//! # Example
//!
//! ```ignore
//! use thundercube::streaming::{AsyncStream, GpuPtr};
//!
//! let stream = AsyncStream::new();
//!
//! // Get raw pointers BEFORE launching blocking kernel
//! let data_ptr: GpuPtr<f32> = stream.ptr(&client, &data_handle);
//! let ctrl_ptr: GpuPtr<u32> = stream.ptr(&client, &ctrl_handle);
//!
//! // Launch your persistent kernel...
//! my_kernel::launch(...);
//!
//! // Now you can read/write while the kernel runs
//! stream.write(ctrl_ptr, &[1, 0, 0]);
//!
//! loop {
//!     let data = stream.read(ctrl_ptr, 3);
//!
//!     do_stuff(data);
//! }
//!
//! let result = stream.read(data_ptr, n);
//! ```

use std::marker::PhantomData;

use bytemuck::Pod;
use cubecl::prelude::*;
use cubecl_hip_sys::{
    HIP_SUCCESS, hipDeviceptr_t, hipMemcpyAsync, hipMemcpyKind_hipMemcpyDeviceToHost,
    hipMemcpyKind_hipMemcpyHostToDevice, hipStream_t, hipStreamCreate, hipStreamDestroy,
    hipStreamSynchronize,
};

/// Raw GPU pointer for direct memory access.
///
/// Obtained via [`AsyncStream::ptr`]. The pointer remains valid as long as
/// the underlying cubecl handle is alive.
///
/// Tracks buffer capacity for bounds checking on read/write operations.
#[derive(Clone, Copy)]
pub struct GpuPtr<'a, T> {
    ptr: hipDeviceptr_t,
    /// Number of elements of type T
    len: usize,
    _marker: PhantomData<&'a T>,
}

impl<T> GpuPtr<'_, T> {
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn size_bytes(&self) -> usize {
        self.len * std::mem::size_of::<T>()
    }
}

/// Mirror of cubecl_hip's internal GpuResource layout.
/// Used to extract the raw pointer from a cubecl binding.
#[repr(C)]
struct GpuResourceCompat {
    ptr: hipDeviceptr_t,
    _binding: *mut std::ffi::c_void,
    _size: u64,
}

/// Async I/O stream for communicating with running kernels.
///
/// Creates a separate HIP stream that doesn't synchronize with the default
/// stream, allowing memory transfers while kernels run.
pub struct AsyncStream {
    stream: hipStream_t,
}

impl AsyncStream {
    pub fn new() -> Self {
        let mut stream: hipStream_t = std::ptr::null_mut();
        unsafe {
            let status = hipStreamCreate(&mut stream);
            if status != HIP_SUCCESS {
                panic!("hipStreamCreate failed with status: {}", status);
            }
        }
        Self { stream }
    }

    /// Extract a raw GPU pointer from a cubecl handle.
    ///
    /// Must be called BEFORE launching a blocking kernel, as this
    /// operation may synchronize with the default stream.
    ///
    /// It is the callers responsibility to ensure that the type
    /// makes sense for the given handle.
    pub fn ptr<'a, T: Pod, R: Runtime>(
        &self,
        client: &'_ ComputeClient<R>,
        handle: &'a cubecl::server::Handle,
    ) -> GpuPtr<'a, T> {
        let binding = handle.clone().binding();
        let resource = client.get_resource(binding);
        let gpu_resource: &GpuResourceCompat =
            unsafe { &*(resource.resource() as *const _ as *const GpuResourceCompat) };
        let size_bytes = gpu_resource._size as usize;
        let elem_size = std::mem::size_of::<T>();
        assert!(
            size_bytes.is_multiple_of(elem_size),
            "buffer size {} is not a multiple of element size {}",
            size_bytes,
            elem_size
        );
        GpuPtr {
            ptr: gpu_resource.ptr,
            len: size_bytes / elem_size,
            _marker: PhantomData,
        }
    }

    pub fn write<T: Pod>(&self, dst: GpuPtr<T>, offset: usize, data: &[T]) {
        assert!(
            offset + data.len() <= dst.len,
            "write at offset {} of {} elements exceeds buffer capacity of {}",
            offset,
            data.len(),
            dst.len
        );
        let size_bytes = std::mem::size_of_val(data);
        let offset_bytes = offset * std::mem::size_of::<T>();
        unsafe {
            let status = hipMemcpyAsync(
                dst.ptr.byte_add(offset_bytes),
                data.as_ptr() as *const std::ffi::c_void,
                size_bytes,
                hipMemcpyKind_hipMemcpyHostToDevice,
                self.stream,
            );
            if status != HIP_SUCCESS {
                panic!("hipMemcpyAsync H2D failed with status: {}", status);
            }
            let status = hipStreamSynchronize(self.stream);
            if status != HIP_SUCCESS {
                panic!("hipStreamSynchronize failed with status: {}", status);
            }
        }
    }

    pub fn read<T: Pod + Clone>(&self, src: GpuPtr<T>, offset: usize, len: usize) -> Vec<T> {
        assert!(
            offset + len <= src.len,
            "read at offset {} of {} elements exceeds buffer capacity of {}",
            offset,
            len,
            src.len
        );
        let mut data = vec![T::zeroed(); len];
        let size_bytes = std::mem::size_of::<T>() * len;
        let offset_bytes = offset * std::mem::size_of::<T>();
        unsafe {
            let status = hipMemcpyAsync(
                data.as_mut_ptr() as *mut std::ffi::c_void,
                src.ptr.byte_add(offset_bytes),
                size_bytes,
                hipMemcpyKind_hipMemcpyDeviceToHost,
                self.stream,
            );
            if status != HIP_SUCCESS {
                panic!("hipMemcpyAsync D2H failed with status: {}", status);
            }
            let status = hipStreamSynchronize(self.stream);
            if status != HIP_SUCCESS {
                panic!("hipStreamSynchronize failed with status: {}", status);
            }
        }
        data
    }
}

impl Default for AsyncStream {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for AsyncStream {
    fn drop(&mut self) {
        unsafe {
            hipStreamDestroy(self.stream);
        }
    }
}
