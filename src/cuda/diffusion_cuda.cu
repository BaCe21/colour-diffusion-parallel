// FILE: flood_sim.cu
// CUDA 12.6 compatible implementation of a color diffusion simulation using a 5- or 9-point
// stencil. Uses shared memory (tiling) to accelerate neighbor loads. Produces PPM frames
// that can be turned into an animation with the provided Python script.

// Build: nvcc -std=c++17 -O3 flood_sim.cu -o flood_sim
// Run example: ./flood_sim --w 1000 --h 1000 --steps 1000 --stencil 9 --tile 32 --save_every 10 --outdir frames

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <string>
#include <chrono>
#include <filesystem>
// removed getopt.h for Windows compatibility

#define CHECK_CUDA(call) do { cudaError_t e = (call); if(e!=cudaSuccess){fprintf(stderr,"CUDA %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); exit(1);}} while(0)

// We store colors as float3 (r,g,b) in device memory
struct float3f { float x, y, z; };

// Kernel parameters: tile (block) handles TILE x TILE interior pixels, using shared memory including halo
// Shared mem layout: (TILE + 2*HALO) x (TILE + 2*HALO)

// Compute average of neighbors depending on stencil
__device__ __forceinline__ float3f avg_stencil_5(const float3f* s, int sx, int x, int y) {
    // central, up, down, left, right
    float3f c = s[y * sx + x];
    float3f up = s[(y - 1) * sx + x];
    float3f down = s[(y + 1) * sx + x];
    float3f left = s[y * sx + (x - 1)];
    float3f right = s[y * sx + (x + 1)];
    float3f res;
    res.x = (c.x + up.x + down.x + left.x + right.x) * (1.0f / 5.0f);
    res.y = (c.y + up.y + down.y + left.y + right.y) * (1.0f / 5.0f);
    res.z = (c.z + up.z + down.z + left.z + right.z) * (1.0f / 5.0f);
    return res;
}

__device__ __forceinline__ float3f avg_stencil_9(const float3f* s, int sx, int x, int y) {
    // 3x3 average including diagonals (9 points)
    float3f sum = { 0.f,0.f,0.f };
    for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
            float3f v = s[(y + dy) * sx + (x + dx)];
            sum.x += v.x; sum.y += v.y; sum.z += v.z;
        }
    }
    sum.x *= (1.0f / 9.0f); sum.y *= (1.0f / 9.0f); sum.z *= (1.0f / 9.0f);
    return sum;
}

// Kernel: input -> output using shared memory tile with halo
// width, height: global dimensions
// tile: interior tile size (without halos). halo = 1 for 5/9 point stencils
// stencil = 5 or 9

extern "C" __global__ void diffuse_kernel(const float3f* in, float3f* out, int width, int height, int tile, int stencil) {
    const int halo = 1;
    const int sx = tile + 2 * halo;
    extern __shared__ float3f s_mem[]; // size: sx * (tile + 2*halo)

    int bx = blockIdx.x; int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;

    // global coords of the interior pixel handled by this thread
    int gx = bx * tile + tx; // tx in [0, tile-1] expected, but block may be larger; we will guard
    int gy = by * tile + ty;

    // shared coords include halo offset
    int sx_off = tx + halo;
    int sy_off = ty + halo;

    // For simplicity, set blockDim to (tile, tile). If threads outside interior exist, they will be ignored.

    // Load interior and halo
    // Each thread loads one interior cell and also some threads load halo neighbors. A simple approach:
    // Each thread loads its cell into shared mem at (sx_off, sy_off). Then we separately have threads at borders load halos.

    // Compute global clamped coordinate helper
    auto clamp = [&](int v, int hi) { if (v < 0) return 0; if (v > hi - 1) return hi - 1; return v; };

    // Load central value
    if (tx < tile && ty < tile) {
        int cx = clamp(gx, width);
        int cy = clamp(gy, height);
        s_mem[sy_off * sx + sx_off] = in[cy * width + cx];
    }

    __syncthreads();

    // Load halos: threads on edges of tile load needed halo values
    // Left halo
    if (tx < tile && ty < tile) {
        if (tx == 0) {
            int hx = clamp(gx - 1, width);
            int hy = clamp(gy, height);
            s_mem[sy_off * sx + (sx_off - 1)] = in[hy * width + hx];
        }
        if (tx == tile - 1) {
            int hx = clamp(gx + 1, width);
            int hy = clamp(gy, height);
            s_mem[sy_off * sx + (sx_off + 1)] = in[hy * width + hx];
        }
        if (ty == 0) {
            int hx = clamp(gx, width);
            int hy = clamp(gy - 1, height);
            s_mem[(sy_off - 1) * sx + sx_off] = in[hy * width + hx];
        }
        if (ty == tile - 1) {
            int hx = clamp(gx, width);
            int hy = clamp(gy + 1, height);
            s_mem[(sy_off + 1) * sx + sx_off] = in[hy * width + hx];
        }
        // corners for 9-point
        if (stencil == 9) {
            if (tx == 0 && ty == 0) {
                int hx = clamp(gx - 1, width); int hy = clamp(gy - 1, height);
                s_mem[(sy_off - 1) * sx + (sx_off - 1)] = in[hy * width + hx];
            }
            if (tx == 0 && ty == tile - 1) {
                int hx = clamp(gx - 1, width); int hy = clamp(gy + 1, height);
                s_mem[(sy_off + 1) * sx + (sx_off - 1)] = in[hy * width + hx];
            }
            if (tx == tile - 1 && ty == 0) {
                int hx = clamp(gx + 1, width); int hy = clamp(gy - 1, height);
                s_mem[(sy_off - 1) * sx + (sx_off + 1)] = in[hy * width + hx];
            }
            if (tx == tile - 1 && ty == tile - 1) {
                int hx = clamp(gx + 1, width); int hy = clamp(gy + 1, height);
                s_mem[(sy_off + 1) * sx + (sx_off + 1)] = in[hy * width + hx];
            }
        }
    }

    __syncthreads();

    // Now compute average for interior pixels only
    if (tx < tile && ty < tile && gx < width && gy < height) {
        float3f res;
        if (stencil == 5) res = avg_stencil_5(s_mem, sx, sx_off, sy_off);
        else res = avg_stencil_9(s_mem, sx, sx_off, sy_off);
        out[gy * width + gx] = res;
    }
}

// Host helpers: initialize grid with a few color sources
void place_sources(std::vector<float3f>& buf, int W, int H) {
    // Clear
    for (int i = 0; i < W * H; ++i) { buf[i].x = buf[i].y = buf[i].z = 0.f; }
    // Place a few colored blobs
    auto put = [&](int cx, int cy, float r, float g, float b, int radius) {
        for (int y = cy - radius; y <= cy + radius; ++y) for (int x = cx - radius; x <= cx + radius; ++x) {
            if (x < 0 || x >= W || y < 0 || y >= H) continue;
            int dx = x - cx, dy = y - cy; if (dx * dx + dy * dy > radius * radius) continue;
            float f = 1.0f - sqrtf((float)(dx * dx + dy * dy)) / (float)radius;
            int idx = y * W + x;
            buf[idx].x = r * f; buf[idx].y = g * f; buf[idx].z = b * f;
        }
        };
    put(W / 4, H / 3, 1.f, 0.f, 0.f, 30);
    put(3 * W / 4, 2 * H / 3, 0.f, 1.f, 0.f, 30);
    put(W / 2, H / 2, 0.f, 0.f, 1.f, 40);
}

void write_ppm(const std::string& path, const std::vector<float3f>& buf, int W, int H) {
    FILE* f = fopen(path.c_str(), "wb");
    if (!f) { perror("fopen"); return; }
    fprintf(f, "P6\n%d %d\n255\n", W, H);
    for (int i = 0; i < W * H; ++i) {
        unsigned char r = (unsigned char)(fminf(1.f, buf[i].x) * 255.0f);
        unsigned char g = (unsigned char)(fminf(1.f, buf[i].y) * 255.0f);
        unsigned char b = (unsigned char)(fminf(1.f, buf[i].z) * 255.0f);
        fwrite(&r, 1, 1, f); fwrite(&g, 1, 1, f); fwrite(&b, 1, 1, f);
    }
    fclose(f);
}

void parse_args(int argc, char** argv,
    int& W, int& H, int& steps, int& stencil, int& tile,
    int& save_every, std::string& outdir)
{
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--w") && i + 1 < argc) W = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--h") && i + 1 < argc) H = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--steps") && i + 1 < argc) steps = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--stencil") && i + 1 < argc) stencil = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--tile") && i + 1 < argc) tile = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--save_every") && i + 1 < argc) save_every = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--outdir") && i + 1 < argc) outdir = argv[++i];
    }
}

int main(int argc, char** argv) {
    int W = 1000, H = 1000, steps = 1000;
    int stencil = 9;
    int tile = 32; // interior tile size used for blockDim
    int save_every = 10;
    std::string outdir = "frames";

    // Simple arg parse (no getopt)
    parse_args(argc, argv, W, H, steps, stencil, tile, save_every, outdir);

    size_t N = (size_t)W * H;
    std::vector<float3f> host_buf(N);
    place_sources(host_buf, W, H);

    float3f* d_a = nullptr, * d_b = nullptr;
    CHECK_CUDA(cudaMalloc(&d_a, N * sizeof(float3f)));
    CHECK_CUDA(cudaMalloc(&d_b, N * sizeof(float3f)));
    CHECK_CUDA(cudaMemcpy(d_a, host_buf.data(), N * sizeof(float3f), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(d_b, 0, N * sizeof(float3f)));

    dim3 block(tile, tile);
    dim3 grid((W + tile - 1) / tile, (H + tile - 1) / tile);

    // shared memory size: (tile+2)^2 * sizeof(float3f)
    int sx = tile + 2;
    size_t smem = (size_t)sx * (tile + 2) * sizeof(float3f);

    auto t0 = std::chrono::high_resolution_clock::now();
    for (int s = 0; s < steps; ++s) {
        diffuse_kernel << <grid, block, smem >> > (d_a, d_b, W, H, tile, stencil);
        CHECK_CUDA(cudaGetLastError());
        // swap
        std::swap(d_a, d_b);

        if ((s % save_every) == 0) {
            // copy back and write ppm
            CHECK_CUDA(cudaMemcpy(host_buf.data(), d_a, N * sizeof(float3f), cudaMemcpyDeviceToHost));
            char path[1024]; sprintf(path, "%s/frame_%06d.ppm", outdir.c_str(), s);
            write_ppm(path, host_buf, W, H);
        }
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double secs = std::chrono::duration<double>(t1 - t0).count();
    printf("Done %d steps on %dx%d in %.3f s\n", steps, W, H, secs);

    // final frame
    CHECK_CUDA(cudaMemcpy(host_buf.data(), d_a, N * sizeof(float3f), cudaMemcpyDeviceToHost));
    write_ppm(outdir + "/frame_final.ppm", host_buf, W, H);

    cudaFree(d_a); cudaFree(d_b);
    return 0;
}
