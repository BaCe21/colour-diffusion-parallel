# Parallel Color Diffusion Simulation
C++ parallel-computing project implementing the same 2D color diffusion workload using sequential execution, OpenMP, MPI and CUDA.
The project was developed as a university team project to compare different parallelization models and their performance characteristics.
## Simulation
The simulation operates on a 2D RGB grid.
Each iteration applies a 3×3 stencil to every cell, averaging the values of the surrounding neighborhood. Repeating this operation produces a diffusion-like spreading effect from several initial color sources.
The implementations use consistent clamped boundary handling to make performance comparisons more meaningful.
## Implementations
### Sequential
Single-threaded C++ implementation used as the performance baseline.
### OpenMP
Shared-memory parallel implementation using OpenMP.
Rows of the grid are processed in parallel using a static work schedule.
### MPI
Distributed-memory implementation that partitions the grid into horizontal regions.
Neighboring processes exchange halo rows using MPI_Sendrecv, while MPI_Gatherv is used when full output frames need to be reconstructed.
### CUDA
GPU implementation using CUDA and shared-memory tiling.
The CUDA portion was developed collaboratively as part of the team project.
### Benchmarking
The repository includes tooling for comparing execution time across:
- sequential execution
- multiple OpenMP thread counts
- multiple MPI process counts
- CUDA execution
Benchmark runs disable frame generation to reduce file-I/O influence on timing results.
Generated results can be processed with the included Python scripts to create comparison charts.
### Visualization
Simulation frames can optionally be exported as PPM images.
A Python utility converts generated frames into animated GIFs for visual inspection of the diffusion process.
### Tech Stack
- C++17
- OpenMP
- MPI
- CUDA
- CMake
- Python
- Pandas
- Matplotlib
- Pillow
### Project Structure
src/seq — sequential baseline
src/openmp — OpenMP implementation
src/mpi — MPI implementation
src/cuda — CUDA implementation and benchmark tooling
### Build
The project uses CMake.
Typical workflow:
cmake -S . -B build
cmake --build build --config Release
MPI and OpenMP are required.
The CUDA target is built only when a compatible CUDA Toolkit installation is detected.
### Example
Example MPI execution:
mpiexec -n 4 mpi_exec --w 1000 --h 1000 --steps 5000 --save_every 0
### Project Status
This repository is preserved as a parallel-computing and performance-analysis portfolio project.
It demonstrates the use of shared-memory, distributed-memory and GPU parallelization approaches on the same grid-based workload.