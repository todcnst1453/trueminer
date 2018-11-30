
cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
set(CMAKE_CUDA_COMPILER "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v9.2/bin/nvcc.exe" )
project(cmake_and_cuda LANGUAGES CXX CUDA)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode=arch=compute_61,code=sm_61 -maxrregcount=60")

add_executable(ChainHash 
    fchainhash.cu
    RefHash.cu
    sha3.cu
    tbl.cu
    hash.h
    gpu_var.h
    sha3.h
    CuAPI_helper.h
    )

set_target_properties(ChainHash PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(ChainHash cuda)

