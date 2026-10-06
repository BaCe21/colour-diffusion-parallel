cmake_minimum_required(VERSION 3.20)
project(colour_diffusion LANGUAGES C CXX) 

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Release)

find_package(MPI REQUIRED)
if(MPI_CXX_FOUND)
  message(STATUS "MPI found: ${MPI_CXX_COMPILE_FLAGS} ${MPI_CXX_LINK_FLAGS}")
endif()

find_package(OpenMP REQUIRED)
if (OpenMP_CXX_FOUND)
  message(STATUS "OpenMP CXX flags: ${OpenMP_CXX_FLAGS}")
endif()

find_package(CUDAToolkit QUIET)
if (CUDAToolkit_FOUND)
  message(STATUS "CUDAToolkit found: ${CUDAToolkit_VERSION}")
  enable_language(CUDA)
  set(HAVE_CUDA TRUE)
else()
  message(STATUS "CUDAToolkit NOT found — CUDA target will be skipped.")
  set(HAVE_CUDA FALSE)
endif()

add_executable(openmp_exec src/openmp/main_openmp.cpp)
if (OpenMP_CXX_FOUND)
  target_compile_options(openmp_exec PRIVATE ${OpenMP_CXX_FLAGS})
endif()
target_include_directories(openmp_exec PRIVATE src/common)

add_executable(mpi_exec src/mpi/main_mpi.cpp)
target_include_directories(mpi_exec PRIVATE src/common)
target_link_libraries(mpi_exec PRIVATE MPI::MPI_CXX)

if (HAVE_CUDA)
  add_executable(cuda_exec src/cuda/diffusion_cuda.cu)

  set_target_properties(
    cuda_exec
    PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
  )

  target_include_directories(
    cuda_exec
    PRIVATE
    src/common
  )

  target_link_libraries(
    cuda_exec
    PRIVATE
    CUDA::cudart
  )
endif()
