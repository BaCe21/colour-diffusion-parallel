// src/openmp/main_openmp.cpp
// Simple 2D color diffusion (prototype) using OpenMP.
// Writes final frame(s) as binary PPM (P6).
// Build: (via CMake) or compile manually with: cl /O2 /openmp main_openmp.cpp

#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <chrono>
#include <cmath>
#include <cstring>
#include <omp.h>
#include <algorithm>  
#include <iomanip>    

struct Pixel { float r, g, b; };

static inline void clamp01(float &x) { if (x < 0.f) x = 0.f; if (x > 1.f) x = 1.f; }

void save_ppm(const std::string &filename, const std::vector<Pixel>& grid, int w, int h) {
    std::ofstream f(filename, std::ios::binary);
    f << "P6\n" << w << " " << h << "\n255\n";
    for (int i = 0; i < w*h; ++i) {
        unsigned char r = (unsigned char)std::lround(std::max(0.f, std::min(1.f, grid[i].r)) * 255.0f);
        unsigned char g = (unsigned char)std::lround(std::max(0.f, std::min(1.f, grid[i].g)) * 255.0f);
        unsigned char b = (unsigned char)std::lround(std::max(0.f, std::min(1.f, grid[i].b)) * 255.0f);
        f.put((char)r); f.put((char)g); f.put((char)b);
    }
    f.close();
}

int main(int argc, char** argv) {
    // Default parameters
    int N = 1000;
    int iters = 500;
    int save_interval = 0; // 0 = only final
    int threads = 0;       // 0 => use OMP env or default
    int stencil = 5;       // 5 or 9

    // parse args simple: --size, --iters, --save, --threads, --stencil
    for (int i=1;i<argc;i++){
        if (strcmp(argv[i], "--size")==0 && i+1<argc) N = atoi(argv[++i]);
        else if (strcmp(argv[i], "--iters")==0 && i+1<argc) iters = atoi(argv[++i]);
        else if (strcmp(argv[i], "--save")==0 && i+1<argc) save_interval = atoi(argv[++i]);
        else if (strcmp(argv[i], "--threads")==0 && i+1<argc) threads = atoi(argv[++i]);
        else if (strcmp(argv[i], "--stencil")==0 && i+1<argc) stencil = atoi(argv[++i]);
        else if (strcmp(argv[i], "--help")==0) {
            std::cout << "Usage: --size N --iters K --save interval --threads T --stencil 5|9\n";
            return 0;
        }
    }

    if (threads > 0) omp_set_num_threads(threads);
    int used_threads = omp_get_max_threads();

    std::cout << "Grid: " << N << "x" << N << ", iters=" << iters
              << ", stencil=" << stencil << ", threads=" << used_threads
              << ", save_interval=" << save_interval << "\n";

    std::vector<Pixel> grid(N * N);
    std::vector<Pixel> newgrid(N * N);

    // initialize: background black (0,0,0), add a few color sources
    auto set_px = [&](int x, int y, float r, float g, float b){
        if (x>=0 && x<N && y>=0 && y<N) grid[y*N + x].r = r, grid[y*N + x].g = g, grid[y*N + x].b = b;
    };

    // sources: center red, top-left green, bottom-right blue
    set_px(N/2, N/2, 1.f, 0.f, 0.f);
    set_px(N/4, N/4, 0.f, 1.f, 0.f);
    set_px(3*N/4, 3*N/4, 0.f, 0.f, 1.f);

    // small disk sources to make effect visible
    int radius = std::max(1, N/100);
    for (int dy=-radius; dy<=radius; ++dy)
    for (int dx=-radius; dx<=radius; ++dx) {
        if (dx*dx + dy*dy <= radius*radius) {
            set_px(N/2+dx, N/2+dy, 1.f, 0.f, 0.f);
            set_px(N/4+dx, N/4+dy, 0.f, 1.f, 0.f);
            set_px(3*N/4+dx, 3*N/4+dy, 0.f, 0.f, 1.f);
        }
    }

    auto t0 = std::chrono::high_resolution_clock::now();

    // main iteration loop
    for (int it=0; it<iters; ++it) {
        // interior only; keep border fixed (zero) for simplicity
        #pragma omp parallel for schedule(static)
        for (int y=1; y<N-1; ++y) {
            int base = y * N;
            for (int x=1; x<N-1; ++x) {
                Pixel acc{0.f,0.f,0.f};
                float count = 0.f;

                // include center
                acc.r += grid[base + x].r;
                acc.g += grid[base + x].g;
                acc.b += grid[base + x].b;
                count += 1.f;

                // 4-neighbors
                acc.r += grid[base + (x-1)].r; acc.g += grid[base + (x-1)].g; acc.b += grid[base + (x-1)].b; count += 1.f;
                acc.r += grid[base + (x+1)].r; acc.g += grid[base + (x+1)].g; acc.b += grid[base + (x+1)].b; count += 1.f;
                acc.r += grid[(y-1)*N + x].r; acc.g += grid[(y-1)*N + x].g; acc.b += grid[(y-1)*N + x].b; count += 1.f;
                acc.r += grid[(y+1)*N + x].r; acc.g += grid[(y+1)*N + x].g; acc.b += grid[(y+1)*N + x].b; count += 1.f;

                if (stencil == 9) {
                    // add diagonals
                    acc.r += grid[(y-1)*N + (x-1)].r; acc.g += grid[(y-1)*N + (x-1)].g; acc.b += grid[(y-1)*N + (x-1)].b; count += 1.f;
                    acc.r += grid[(y-1)*N + (x+1)].r; acc.g += grid[(y-1)*N + (x+1)].g; acc.b += grid[(y-1)*N + (x+1)].b; count += 1.f;
                    acc.r += grid[(y+1)*N + (x-1)].r; acc.g += grid[(y+1)*N + (x-1)].g; acc.b += grid[(y+1)*N + (x-1)].b; count += 1.f;
                    acc.r += grid[(y+1)*N + (x+1)].r; acc.g += grid[(y+1)*N + (x+1)].g; acc.b += grid[(y+1)*N + (x+1)].b; count += 1.f;
                }

                newgrid[base + x].r = acc.r / count;
                newgrid[base + x].g = acc.g / count;
                newgrid[base + x].b = acc.b / count;
            }
        }

        // swap buffers
        std::swap(grid, newgrid);

        // optional: save intermediate frames
        if (save_interval > 0 && ((it+1) % save_interval == 0)) {
            std::ostringstream ss; ss << "frame_" << std::setfill('0') << std::setw(4) << (it+1) << ".ppm";
            save_ppm(ss.str(), grid, N, N);
        }
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = t1 - t0;

    std::cout << "Done. Time: " << elapsed.count() << " s\n";
    double cells = double(N) * double(N) * double(iters);
    std::cout << "Throughput: " << (cells / elapsed.count()) / 1e6 << " Mcells/s\n";

    // save final image
    save_ppm("final.ppm", grid, N, N);
    std::cout << "Wrote final.ppm\n";
    return 0;
}
