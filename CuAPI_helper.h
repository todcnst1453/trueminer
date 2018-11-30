#pragma  once
#include "cuda_runtime.h"
#include <iostream>
#include <memory>
#include <map>
#include <stdexcept>
#include <vector>
#include <algorithm>

#define CUDA_GET_SYMBOL_ADDR(devPtr, sym)                   \
{                                                           \
    devPtr = nullptr;                                       \
    auto cudaStatus = cudaGetSymbolAddress(&devPtr, sym);   \
    if (cudaStatus != cudaSuccess) {                        \
        throw std::runtime_error("Error return code");      \
    }                                                       \
}

class CuChecker
{
public:
    CuChecker(cudaError_t err)
    {
        Check(err);
    }

    CuChecker& operator=(cudaError_t err)
    {
        Check(err);
        return *this;
    }

    bool operator!=(cudaError_t err)
    {
        return true;
    }

private:
    void Check(cudaError_t err)
    {
        if (err != cudaSuccess) {
            std::cerr << "Cuda runtime error " << cudaGetErrorString(err) << std::endl;;
            throw std::runtime_error("Cuda run time error");
        }
    }
};


class CuDeviceVarManager
{
public:
    struct VarInfo
    {
        size_t typeHash;
        size_t arrayLength; // 0 means it is not an array
        size_t elementSize;
        void* devPtr;
    };

    static CuDeviceVarManager& GetInstance()
    {
        static CuDeviceVarManager s_mgr;
        return s_mgr;
    }

    template<typename T>
    const VarInfo& AddVar(const char* varName, void* devPtr, size_t arrayLength = 0)
    {
        if (!devPtr) {
            throw std::runtime_error("Invalid GPU ptr");
        }
        VarInfo info = { typeid(T).hash_code(), arrayLength, sizeof(T), devPtr };
        m_varMap[varName] = info;

        return m_varMap[varName];
    }

    const VarInfo& GetVar(const char* varName)
    {
        auto it = m_varMap.find(varName);
        if (it == m_varMap.end()) {
            throw std::runtime_error("Can't find var in the map.");
        }
        return it->second;
    }

    template<typename T>
    void CopyToGpuSingleVal(const VarInfo& var, T val)
    {
        if (var.typeHash != typeid(T).hash_code()) {
            throw std::runtime_error("Type mismatch.");
        }

        if (var.arrayLength) {
            throw std::runtime_error("Var is array type. You can't set single value");
        }

        CuChecker cuStatus = cudaMemcpy(var.devPtr, &val, sizeof(T), cudaMemcpyHostToDevice);
    }

    template<typename T>
    void GetFromGpuSingleVal(const VarInfo& var, T& val)
    {
        if (var.typeHash != typeid(T).hash_code()) {
            throw std::runtime_error("Type mismatch.");
        }

        if (var.arrayLength) {
            throw std::runtime_error("Var is array type. You can't get single value");
        }

        CuChecker cuStatus = cudaMemcpy(&val, var.devPtr, sizeof(T), cudaMemcpyDeviceToHost);
    }

    template<typename T>
    T GetFromGpuSingleVal(const VarInfo& var)
    {
        T ret;

        if (var.typeHash != typeid(T).hash_code()) {
            throw std::runtime_error("Type mismatch.");
        }

        if (var.arrayLength) {
            throw std::runtime_error("Var is array type. You can't get single value");
        }

        CuChecker cuStatus = cudaMemcpy(&ret, var.devPtr, sizeof(T), cudaMemcpyDeviceToHost);

        return ret;
    }

    template<typename T>
    void CopyToGpuArray(const VarInfo& var, T* ptr, size_t len)
    {
        if (var.typeHash != typeid(T).hash_code()) {
            throw std::runtime_error("Type mismatch.");
        }

        if (!var.arrayLength) {
            throw std::runtime_error("Var is single value. You can't copy array");
        }

        if (var.arrayLength != len) {
            throw std::runtime_error("Array size mismatch");
        }

        CuChecker cuStatus = cudaMemcpy(var.devPtr, ptr, sizeof(T) * len, cudaMemcpyHostToDevice);
    }

    template<typename T>
    void GetFromGpuArray(const VarInfo& var, T* ptr, size_t len)
    {
        if (var.typeHash != typeid(T).hash_code()) {
            throw std::runtime_error("Type mismatch.");
        }

        if (!var.arrayLength) {
            throw std::runtime_error("Var is single value. You can't get array");
        }

        if (var.arrayLength != len) {
            throw std::runtime_error("Array size mismatch");
        }

        CuChecker cuStatus = cudaMemcpy(ptr, var.devPtr, sizeof(T) * len, cudaMemcpyDeviceToHost);
    }

    template<typename T>
    std::shared_ptr<std::vector<T>> GetFromGpuArray(const VarInfo& var)
    {
        if (var.typeHash != typeid(T).hash_code()) {
            throw std::runtime_error("Type mismatch.");
        }

        if (!var.arrayLength) {
            throw std::runtime_error("Var is single value. You can't get array");
        }

        auto ret = std::make_shared<std::vector<T>>();
        ret->reserve(var.arrayLength);
        ret->resize(var.arrayLength);

        CuChecker cuStatus = cudaMemcpy(&(*ret)[0], var.devPtr, sizeof(T) * var.arrayLength, cudaMemcpyDeviceToHost);

        return ret;
    }

    void Memset(const VarInfo& var, char val)
    {
        CuChecker cudStats = cudaMemset(var.devPtr, val, var.elementSize * std::max<size_t>(1, var.arrayLength));
    }

private:
    CuDeviceVarManager()
    {
    }

    std::map<std::string, VarInfo> m_varMap;

};