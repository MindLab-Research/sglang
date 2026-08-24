// nvlink_allocator_approach_a.cpp
// 改法 A: 跳过 fabric probe, 用 hipMalloc + ibv_reg_mr (dlopen 动态加载)
// 不 include infiniband/verbs.h, 用 void* 代替 ibv 类型

#include "cuda_alike.h"
#include <sys/types.h>
#include <iostream>
#include <dlfcn.h>
#include <string.h>
#include <errno.h>
#include <hip/hip_runtime.h>

// 前向声明 ibv 类型 (不 include verbs.h)
struct ibv_device;
struct ibv_context;
struct ibv_pd;
struct ibv_mr;

// ibv 函数指针类型
typedef struct ibv_device **(*ibv_get_device_list_t)(int *);
typedef struct ibv_context *(*ibv_open_device_t)(struct ibv_device *);
typedef struct ibv_pd *(*ibv_alloc_pd_t)(struct ibv_context *);
typedef struct ibv_mr *(*ibv_reg_mr_t)(struct ibv_pd *, void *, size_t, int);
typedef int (*ibv_dereg_mr_t)(struct ibv_mr *);
typedef int (*ibv_close_device_t)(struct ibv_context *);
typedef void (*ibv_free_device_list_t)(struct ibv_device **);

// IBV access flags
#define MY_IBV_ACCESS_LOCAL_WRITE   1
#define MY_IBV_ACCESS_REMOTE_WRITE  (1 << 1)
#define MY_IBV_ACCESS_REMOTE_READ   (1 << 2)
#define MY_IBV_ACCESS_REMOTE_ATOMIC (1 << 3)

struct MRWrapper {
    void *mr_handle;
    void *ptr;
    size_t size;
    ibv_dereg_mr_t dereg_mr;
};

#include <unordered_map>
static std::unordered_map<void*, MRWrapper*> g_mr_registry;
static pthread_mutex_t g_mr_lock = PTHREAD_MUTEX_INITIALIZER;

enum class MemoryBackendType { use_cudamalloc, use_cumemcreate, unknown };

namespace {

MemoryBackendType ProbeAllocatorBackend(int device_id) {
    std::cerr << "[nvlink_allocator] Approach A: skipping fabric probe, using hipMalloc + ibv_reg_mr (dlopen)\n";
    return MemoryBackendType::use_cumemcreate;
}

void *AllocateFabricMemory(ssize_t size, int device, cudaStream_t stream) {
    (void)stream;

    // 1. hipMalloc 分配 GPU 内存
    void *ptr = nullptr;
    hipSetDevice(device);
    hipError_t err = hipMalloc(&ptr, size);
    if (err != hipSuccess) {
        std::cerr << "[nvlink_allocator] hipMalloc failed: " << err << " (size=" << size << ")\n";
        return nullptr;
    }
    std::cerr << "[nvlink_allocator] hipMalloc OK: ptr=" << ptr << " size=" << size << " device=" << device << "\n";

    // 2. dlopen libibverbs
    void *libibverbs = dlopen("libibverbs.so", RTLD_NOW | RTLD_GLOBAL);
    if (!libibverbs) {
        std::cerr << "[nvlink_allocator] dlopen libibverbs.so failed: " << dlerror() << "\n";
        return ptr;
    }

    // 3. 加载 ibv 函数
    auto ibv_get_device_list = (ibv_get_device_list_t)dlsym(libibverbs, "ibv_get_device_list");
    auto ibv_open_device = (ibv_open_device_t)dlsym(libibverbs, "ibv_open_device");
    auto ibv_alloc_pd = (ibv_alloc_pd_t)dlsym(libibverbs, "ibv_alloc_pd");
    auto ibv_reg_mr = (ibv_reg_mr_t)dlsym(libibverbs, "ibv_reg_mr");
    auto ibv_dereg_mr = (ibv_dereg_mr_t)dlsym(libibverbs, "ibv_dereg_mr");
    auto ibv_close_device = (ibv_close_device_t)dlsym(libibverbs, "ibv_close_device");
    auto ibv_free_device_list = (ibv_free_device_list_t)dlsym(libibverbs, "ibv_free_device_list");

    if (!ibv_get_device_list || !ibv_open_device || !ibv_alloc_pd || !ibv_reg_mr || !ibv_dereg_mr) {
        std::cerr << "[nvlink_allocator] dlsym failed for ibv functions\n";
        return ptr;
    }

    // 4. 打开 IB device
    int num_devices = 0;
    struct ibv_device **dev_list = ibv_get_device_list(&num_devices);
    if (!dev_list || num_devices == 0) {
        std::cerr << "[nvlink_allocator] No IB devices found\n";
        return ptr;
    }

    struct ibv_context *ctx = ibv_open_device(dev_list[0]);
    if (!ctx) {
        std::cerr << "[nvlink_allocator] ibv_open_device failed\n";
        ibv_free_device_list(dev_list);
        return ptr;
    }

    // 5. 分配 PD
    struct ibv_pd *pd = ibv_alloc_pd(ctx);
    if (!pd) {
        std::cerr << "[nvlink_allocator] ibv_alloc_pd failed: " << strerror(errno) << "\n";
        ibv_close_device(ctx);
        ibv_free_device_list(dev_list);
        return ptr;
    }

    // 6. 注册 GPU 内存为 MR
    int access_flags = MY_IBV_ACCESS_LOCAL_WRITE | MY_IBV_ACCESS_REMOTE_WRITE | MY_IBV_ACCESS_REMOTE_READ | MY_IBV_ACCESS_REMOTE_ATOMIC;
    struct ibv_mr *mr = ibv_reg_mr(pd, ptr, size, access_flags);
    if (!mr) {
        std::cerr << "[nvlink_allocator] ibv_reg_mr failed: " << strerror(errno) << " (errno=" << errno << ")\n";
        ibv_close_device(ctx);
        ibv_free_device_list(dev_list);
        return ptr;
    }

    std::cerr << "[nvlink_allocator] ibv_reg_mr OK: mr=" << mr << "\n";

    // 7. 保存 MR wrapper
    MRWrapper *wrapper = new MRWrapper{(void*)mr, ptr, (size_t)size, ibv_dereg_mr};
    pthread_mutex_lock(&g_mr_lock);
    g_mr_registry[ptr] = wrapper;
    pthread_mutex_unlock(&g_mr_lock);

    return ptr;
}

void FreeFabricMemory(void *ptr, ssize_t ssize, int device, cudaStream_t stream) {
    (void)ssize; (void)device; (void)stream;
    if (!ptr) return;

    pthread_mutex_lock(&g_mr_lock);
    auto it = g_mr_registry.find(ptr);
    if (it != g_mr_registry.end()) {
        MRWrapper *wrapper = it->second;
        if (wrapper->mr_handle && wrapper->dereg_mr) {
            wrapper->dereg_mr((struct ibv_mr*)wrapper->mr_handle);
            std::cerr << "[nvlink_allocator] ibv_dereg_mr OK: ptr=" << ptr << "\n";
        }
        delete wrapper;
        g_mr_registry.erase(it);
    }
    pthread_mutex_unlock(&g_mr_lock);

    hipFree(ptr);
}

}  // namespace

extern "C" {

MemoryBackendType mc_probe_fabric_support(int device_id) {
    return ProbeAllocatorBackend(device_id);
}

int mc_allocator_probe(int device_id) {
    return static_cast<int>(ProbeAllocatorBackend(device_id));
}

void *mc_allocator_malloc(ssize_t size, int device, cudaStream_t stream) {
    return AllocateFabricMemory(size, device, stream);
}

void *mc_nvlink_malloc(ssize_t size, int device, cudaStream_t stream) {
    return mc_allocator_malloc(size, device, stream);
}

void mc_allocator_free(void *ptr, ssize_t ssize, int device, cudaStream_t stream) {
    FreeFabricMemory(ptr, ssize, device, stream);
}

void mc_nvlink_free(void *ptr, ssize_t ssize, int device, cudaStream_t stream) {
    mc_allocator_free(ptr, ssize, device, stream);
}

}
