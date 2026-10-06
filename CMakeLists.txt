cmake_minimum_required(VERSION 3.20)

project(colour_diffusion LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Sequential
add_executable(
    diffusion_seq
    src/seq/diffusion_seq.cpp
)

# OpenMP
find_package(OpenMP REQUIRED)

add_executable(
    main_openmp
    src/openmp/main_openmp.cpp
)

target_link_libraries(
    main_openmp
    PRIVATE
    OpenMP::OpenMP_CXX
)

# MPI
find_package(MPI REQUIRED)

add_executable(
    main_mpi
    src/mpi/main_mpi.cpp
)

target_link_libraries(
    main_mpi
    PRIVATE
    MPI::MPI_CXX
)

# CUDA - optional
find_package(CUDAToolkit QUIET)

if(CUDAToolkit_FOUND)
    enable_language(CUDA)

    add_executable(
        diffusion_cuda
        src/cuda/diffusion_cuda.cu
    )

    set_target_properties(
        diffusion_cuda
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
    )

    target_link_libraries(
        diffusion_cuda
        PRIVATE
        CUDA::cudart
    )
else()
    message(
        STATUS
        "CUDA Toolkit not found - CUDA target will be skipped."
    )
endif()