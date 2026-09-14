// hip_approach_b.h
// 改法 B: 修改 hip.h, 让 fabric probe 立即返回 0 (不支持)
// 这样 ProbeAllocatorBackend 返回 use_cudamalloc
// allocator 返回 NULL, SGLang 用 cudaMalloc 分配 staging buffer (GPU 内存)
// mooncake RDMA transport 尝试 ibv_reg_mr 注册 GPU 内存
// 如果 amdgpu peer memory 已注册 (ib_uverbs 251 amdgpu), ibv_reg_mr 应该能直接注册 GPU 地址

#pragma once

#include <hip/hip_runtime.h>
#include <string>

const static std::string GPU_PREFIX = "hip:";

// hipify-perl warning: unsupported HIP identifier: cudaMemoryTypeUnregistered
#define cudaMemoryTypeUnregistered hipMemoryTypeUnregistered

// 改法 B: 直接定义 fabric 相关常量为 0 或空, 让 probe 立即返回 use_cudamalloc
// 不再调用 cuDeviceGetAttribute (会挂起)
#define CU_MEM_HANDLE_TYPE_FABRIC hipMemHandleTypePosixFileDescriptor

// 改法 B: 让 fabric supported 检查返回 0
// 原始: #define CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED hipDeviceAttributeVirtualMemoryManagementSupported
// 修改: 使用一个一定返回 0 的 attribute
#define CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED hipDeviceAttributePciBusId  // 总是返回 0 (不是 bool)

// 改法 B: GPU Direct RDMA 也返回 0
#define CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WITH_CUDA_VMM_SUPPORTED hipDeviceAttributePciBusId

#define CUmemFabricHandle void*

// HIP mappings for nvlink_allocator compatibility
#define CUDA_ERROR_NOT_PERMITTED hipErrorNotSupported
