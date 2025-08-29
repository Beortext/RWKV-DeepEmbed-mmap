#include <torch/extension.h>
#include <torch/script.h>
#include <future>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>
#include <fstream>
#include <stdexcept>

// POSIX/Linux Headers
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

#include "nlohmann/json.hpp"

// ============================================================================
// 1. MemoryMappedFile Class (No changes)
// ============================================================================
class MemoryMappedFile {
public:
    MemoryMappedFile(const std::string& path) {
        fd_ = open(path.c_str(), O_RDONLY);
        if (fd_ == -1) throw std::runtime_error("Mmap: Failed to open file: " + path);
        struct stat file_info;
        if (fstat(fd_, &file_info) == -1) { close(fd_); throw std::runtime_error("Mmap: Failed to get file size for: " + path); }
        size_ = file_info.st_size;
        if (size_ > 0) {
            ptr_ = mmap(NULL, size_, PROT_READ, MAP_PRIVATE, fd_, 0);
            if (ptr_ == MAP_FAILED) { close(fd_); throw std::runtime_error("Mmap: mmap failed for: " + path); }
        }
    }
    ~MemoryMappedFile() {
        if (ptr_ != MAP_FAILED && ptr_ != nullptr) munmap(ptr_, size_);
        if (fd_ != -1) close(fd_);
    }
    MemoryMappedFile(const MemoryMappedFile&) = delete;
    MemoryMappedFile& operator=(const MemoryMappedFile&) = delete;
    const char* get_data() const { return static_cast<const char*>(ptr_); }
private:
    int fd_ = -1;
    size_t size_ = 0;
    void* ptr_ = MAP_FAILED;
};


// ============================================================================
// 2. DataManager Singleton (with the fix)
// ============================================================================
using json = nlohmann::json;

struct TensorInfo { int64_t offset; c10::ScalarType dtype; std::vector<int64_t> shape; };

class DataManager {
public:
    static DataManager& getInstance(const std::string& file_path) {
        static std::mutex mutex;
        std::lock_guard<std::mutex> lock(mutex);
        static std::unordered_map<std::string, std::unique_ptr<DataManager>> instances;
        
        if (instances.find(file_path) == instances.end()) {
            // --- CORE FIX HERE ---
            // We cannot use std::make_unique because it's not a friend of this class.
            // We must use `new` directly, as this `getInstance` method has access rights.
            instances[file_path] = std::unique_ptr<DataManager>(new DataManager(file_path));
        }
        return *instances.at(file_path);
    }

    const TensorInfo& getTensorInfo(const std::string& key) const {
        return index_cache_.at(key);
    }

    const char* getMmapBasePtr() const {
        return mmap_file_->get_data();
    }

private:
    DataManager(const std::string& file_path) {
        std::ifstream file(file_path, std::ios::binary | std::ios::ate);
        if (!file) throw std::runtime_error("DataManager: Cannot open file: " + file_path);

        auto file_size = file.tellg();
        if (file_size < 16) throw std::runtime_error("DataManager: File is too small to contain a footer.");
        file.seekg(file_size - std::streamoff(16));
        
        char footer_buffer[16];
        file.read(footer_buffer, 16);
        if (file.gcount() != 16) throw std::runtime_error("DataManager: Failed to read 16-byte footer.");

        int64_t index_offset = *reinterpret_cast<int64_t*>(footer_buffer);
        int64_t index_size = *reinterpret_cast<int64_t*>(footer_buffer + 8);

        std::vector<char> index_buffer(index_size);
        file.seekg(index_offset);
        file.read(index_buffer.data(), index_size);
        if (file.gcount() != index_size) throw std::runtime_error("DataManager: Failed to read full index.");

        auto index_json = json::parse(index_buffer);
        for (auto& [key, val] : index_json.items()) {
            index_cache_[key] = {val["offset"], int_to_dtype(val["dtype"]), val["shape"].get<std::vector<int64_t>>()};
        }

        mmap_file_ = std::make_unique<MemoryMappedFile>(file_path);
    }

    c10::ScalarType int_to_dtype(int d) {
        static const std::vector<c10::ScalarType> dtype_map = { at::kFloat, at::kHalf, at::kBFloat16, at::kLong, at::kInt, at::kByte };
        if (d < 0 || d >= dtype_map.size()) throw std::runtime_error("Invalid dtype integer from index.");
        return dtype_map[d];
    }
    
    std::unique_ptr<MemoryMappedFile> mmap_file_;
    std::unordered_map<std::string, TensorInfo> index_cache_;
};

// ============================================================================
// 3. Pybind11 Interface (No changes)
// ============================================================================
torch::Tensor perform_io_task(const std::string& file_path, const std::string& key, const std::vector<int64_t>& token_ids) {
    auto& manager = DataManager::getInstance(file_path);
    const auto& info = manager.getTensorInfo(key);
    const char* file_base_ptr = manager.getMmapBasePtr();
    const void* tensor_data_ptr = file_base_ptr + info.offset;
    auto options = torch::TensorOptions().dtype(info.dtype).device(torch::kCPU);
    torch::Tensor full_tensor = torch::from_blob(const_cast<void*>(tensor_data_ptr), info.shape, options);
    auto indices = torch::tensor(token_ids, torch::kLong);
    return torch::index_select(full_tensor, 0, indices).clone();
}

class AsyncDataHandle {
public:
    AsyncDataHandle(std::future<torch::Tensor>&& future) : future_(std::move(future)) {}
    torch::Tensor get() { return future_.get(); }
private:
    std::future<torch::Tensor> future_;
};

std::shared_ptr<AsyncDataHandle> trigger_io(
    const std::string& file_path, const std::string& key, const std::vector<int64_t>& token_ids) {
    std::future<torch::Tensor> future = std::async(
        std::launch::async, perform_io_task, file_path, key, token_ids
    );
    return std::make_shared<AsyncDataHandle>(std::move(future));
}

TORCH_LIBRARY(custom, m) {
    m.def("perform_io_task_sync(str file_path, str key, int[] token_ids) -> Tensor", &perform_io_task);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    py::class_<AsyncDataHandle, std::shared_ptr<AsyncDataHandle>>(m, "AsyncDataHandle")
        .def("get", &AsyncDataHandle::get, py::call_guard<py::gil_scoped_release>());
    m.def("trigger_io", &trigger_io, "Trigger async data loading from disk.");
}