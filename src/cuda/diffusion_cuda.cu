#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <string>
#include <chrono>
#include <filesystem>

#define CHECK_CUDA(call) do { cudaError_t e = (call); if(e!=cudaSuccess){fprintf(stderr,"CUDA %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); exit(1);}} while(0)

struct float3f { float x, y, z; };

__device__ __forceinline__ float3f avg_stencil_9(const float3f* s, int sx, int x, int y) {
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

__device__ __forceinline__ int clamp_i(int v, int hi) {
    if (v < 0) return 0;
    if (v >= hi) return hi - 1;
    return v;
}

extern "C" __global__ void diffuse_kernel(const float3f* in, float3f* out, int width, int height, int tile) {
    const int halo = 1;
    const int sx = tile + 2 * halo;
    extern __shared__ float3f s_mem[]; 

    int bx = blockIdx.x; int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;

    int gx = bx * tile + tx; 
    int gy = by * tile + ty;

    int sx_off = tx + halo;
    int sy_off = ty + halo;

    // Load central value
    if (tx < tile && ty < tile) {
        int cx = clamp_i(gx, width);
        int cy = clamp_i(gy, height);
        s_mem[sy_off * sx + sx_off] = in[cy * width + cx];
    }

    __syncthreads();

    // Load halos
    if (tx < tile && ty < tile) {

        if (tx == 0) {
            int hx = clamp_i(gx - 1, width);
            int hy = clamp_i(gy, height);
            s_mem[sy_off * sx + (sx_off - 1)] = in[hy * width + hx];
        }
        if (tx == tile - 1) {
            int hx = clamp_i(gx + 1, width);
            int hy = clamp_i(gy, height);
            s_mem[sy_off * sx + (sx_off + 1)] = in[hy * width + hx];
        }
        if (ty == 0) {
            int hx = clamp_i(gx, width);
            int hy = clamp_i(gy - 1, height);
            s_mem[(sy_off - 1) * sx + sx_off] = in[hy * width + hx];
        }
        if (ty == tile - 1) {
            int hx = clamp_i(gx, width);
            int hy = clamp_i(gy + 1, height);
            s_mem[(sy_off + 1) * sx + sx_off] = in[hy * width + hx];
        }


        if (tx == 0 && ty == 0) {
            int hx = clamp_i(gx - 1, width); int hy = clamp_i(gy - 1, height);
            s_mem[(sy_off - 1) * sx + (sx_off - 1)] = in[hy * width + hx];
        }
        if (tx == 0 && ty == tile - 1) {
            int hx = clamp_i(gx - 1, width); int hy = clamp_i(gy + 1, height);
            s_mem[(sy_off + 1) * sx + (sx_off - 1)] = in[hy * width + hx];
        }
        if (tx == tile - 1 && ty == 0) {
            int hx = clamp_i(gx + 1, width); int hy = clamp_i(gy - 1, height);
            s_mem[(sy_off - 1) * sx + (sx_off + 1)] = in[hy * width + hx];
        }
        if (tx == tile - 1 && ty == tile - 1) {
            int hx = clamp_i(gx + 1, width); int hy = clamp_i(gy + 1, height);
            s_mem[(sy_off + 1) * sx + (sx_off + 1)] = in[hy * width + hx];
        }
    }

    __syncthreads();

    // Now compute average for interior pixels only
    if (tx < tile && ty < tile && gx < width && gy < height) {
        float3f res;
        res = avg_stencil_9(s_mem, sx, sx_off, sy_off);
        out[gy * width + gx] = res;
    }
}

void place_sources(std::vector<float3f>& buf, int W, int H) {
    for (int i = 0; i < W * H; ++i) { buf[i].x = buf[i].y = buf[i].z = 0.f; }

    const float beta = 4.0f;      // edge boost factor (edge ~ beta * center)
    const float intensity = 1.8f; // global amplitude multiplier

    auto put = [&](int cx, int cy, float r, float g, float b, int radius) {
        const int r2 = radius * radius;
        for (int y = cy - radius; y <= cy + radius; ++y) {
            for (int x = cx - radius; x <= cx + radius; ++x) {
                if (x < 0 || x >= W || y < 0 || y >= H) continue;
                int dx = x - cx, dy = y - cy;
                int d2 = dx * dx + dy * dy;
                if (d2 > r2) continue;
                float dist = sqrtf((float)d2);
                float t = dist / (float)radius;

                float w = 1.0f + (beta - 1.0f) * t;

                float f = w * intensity;

                int idx = y * W + x;
                buf[idx].x += r * f; 
                buf[idx].y += g * f;
                buf[idx].z += b * f;
            }
        }
        };

    put(W / 4, H / 3, 1.f, 0.f, 0.f, 150);
    put(3 * W / 4, 2 * H / 3, 0.f, 1.f, 0.f, 150);
    put(W / 2, H / 2, 0.f, 0.f, 1.f, 150);
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
    int& W, int& H, int& steps, int& tile,
    int& save_every, std::string& outdir)
{
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--w") && i + 1 < argc) W = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--h") && i + 1 < argc) H = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--steps") && i + 1 < argc) steps = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--tile") && i + 1 < argc) tile = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--save_every") && i + 1 < argc) save_every = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--outdir") && i + 1 < argc) outdir = argv[++i];
    }
}

int main(int argc, char** argv) {
    int W = 1000, H = 1000, steps = 100000;
    int tile = 32;
    int save_every = 500;
    std::string outdir = "frames";

    parse_args(argc, argv, W, H, steps, tile, save_every, outdir);
    std::filesystem::create_directories(outdir);

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

    size_t smem = (size_t)(tile + 2) * (tile + 2) * sizeof(float3f);

    auto t0 = std::chrono::high_resolution_clock::now();
    for (int s = 0; s < steps; ++s) {
        diffuse_kernel << <grid, block, smem >> > (d_a, d_b, W, H, tile);
        CHECK_CUDA(cudaGetLastError());

        std::swap(d_a, d_b);

        if ((s % save_every) == 0) {
            CHECK_CUDA(cudaMemcpy(host_buf.data(), d_a, N * sizeof(float3f), cudaMemcpyDeviceToHost));
            char path[1024]; sprintf(path, "%s/frame_%06d.ppm", outdir.c_str(), s);
            write_ppm(path, host_buf, W, H);
            double total = 0.0;
            for (size_t i = 0; i < N; ++i) total += host_buf[i].x + host_buf[i].y + host_buf[i].z; printf("step %d total energy: %.6f\n", s, total);
        }
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double secs = std::chrono::duration<double>(t1 - t0).count();
    printf("Done %d steps on %dx%d in %.3f s\n", steps, W, H, secs);

    CHECK_CUDA(cudaMemcpy(host_buf.data(), d_a, N * sizeof(float3f), cudaMemcpyDeviceToHost));
    write_ppm(outdir + "/frame_final.ppm", host_buf, W, H);

    cudaFree(d_a); cudaFree(d_b);
    return 0;
}