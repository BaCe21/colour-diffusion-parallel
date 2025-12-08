#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <chrono>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <omp.h>
#include <filesystem>
#include <numeric>

namespace fs = std::filesystem;

struct float3f { float x, y, z; };

// ---------------------------------------------
// Funkcje pomocnicze
// ---------------------------------------------

void write_ppm(const std::string& path, const std::vector<float3f>& buf, int W, int H) {
    FILE* f = fopen(path.c_str(), "wb");
    if (!f) return;
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
    int& W, int& H, int& steps, int& save_every, std::string& outdir, int& repeat, int& threads)
{
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--w") && i + 1 < argc) W = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--h") && i + 1 < argc) H = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--steps") && i + 1 < argc) steps = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--save_every") && i + 1 < argc) save_every = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--outdir") && i + 1 < argc) outdir = argv[++i];
        else if (!strcmp(argv[i], "--repeat") && i + 1 < argc) repeat = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--threads") && i + 1 < argc) threads = atoi(argv[++i]);
    }
}

// ---------------------------------------------
// Logika Fizyczna (Identyczna jak CUDA)
// ---------------------------------------------

void place_sources(std::vector<float3f>& buf, int W, int H) {
    // Reset
    std::fill(buf.begin(), buf.end(), float3f{0.f, 0.f, 0.f});

    const float beta = 4.0f;
    const float intensity = 1.8f;

    auto put = [&](int cx, int cy, float r, float g, float b, int radius) {
        const int r2 = radius * radius;
        // OpenMP tutaj przyspieszy inicjalizację
        #pragma omp parallel for collapse(2)
        for (int y = cy - radius; y <= cy + radius; ++y) {
            for (int x = cx - radius; x <= cx + radius; ++x) {
                if (x < 0 || x >= W || y < 0 || y >= H) continue;
                
                int dx = x - cx;
                int dy = y - cy;
                int d2 = dx * dx + dy * dy;
                
                if (d2 > r2) continue;

                float dist = sqrtf((float)d2);
                float t = dist / (float)radius;
                float w = 1.0f + (beta - 1.0f) * t;
                float f = w * intensity;

                int idx = y * W + x;
                // Uwaga: przy nakładaniu się źródeł wyścig wątków jest możliwy, 
                // ale rzadki i mało istotny wizualnie. Dla 100% poprawności można użyć atomic,
                // ale tu zostawiamy tak dla szybkości (lub single thread wewnątrz put).
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

// Helper do krawędzi
inline int clamp_i(int v, int max_v) {
    if (v < 0) return 0;
    if (v >= max_v) return max_v - 1;
    return v;
}

// ---------------------------------------------
// Main
// ---------------------------------------------
int main(int argc, char** argv) {
    int W = 1000, H = 1000, steps = 100000;
    int save_every = 500;
    int repeat = 1;
    int threads = 0; // 0 = auto
    std::string outdir = "frames_omp";

    parse_args(argc, argv, W, H, steps, save_every, outdir, repeat, threads);

    // Konfiguracja OpenMP
    if (threads > 0) {
        omp_set_num_threads(threads);
    } else {
        threads = omp_get_max_threads();
    }

    std::cout << "OpenMP Run: " << threads << " threads, Grid: " << W << "x" << H << std::endl;

    if (!fs::exists(outdir)) fs::create_directories(outdir);

    // Bufory
    std::vector<float3f> grid_curr(W * H);
    std::vector<float3f> grid_next(W * H);

    std::vector<double> timings;
    std::string csv_filename = "diffusion_omp_" + std::to_string(threads) + "thr_" +
                               std::to_string(W) + "x" + std::to_string(H) + ".csv";

    // Pętla benchmarkowa
    for (int run = 0; run < repeat; ++run) {
        std::cout << "Run " << (run + 1) << "/" << repeat << "... " << std::flush;

        // Inicjalizacja
        place_sources(grid_curr, W, H);
        grid_next = grid_curr;

        auto t0 = std::chrono::high_resolution_clock::now();

        // Główna pętla symulacji
        for (int s = 0; s < steps; ++s) {
            
            // Równoległe przetwarzanie wierszy
            #pragma omp parallel for schedule(static)
            for (int y = 0; y < H; ++y) {
                for (int x = 0; x < W; ++x) {
                    float sum_x = 0, sum_y = 0, sum_z = 0;

                    // 9-point stencil
                    for (int dy = -1; dy <= 1; ++dy) {
                        for (int dx = -1; dx <= 1; ++dx) {
                            int cy = clamp_i(y + dy, H);
                            int cx = clamp_i(x + dx, W);
                            
                            float3f v = grid_curr[cy * W + cx];
                            sum_x += v.x;
                            sum_y += v.y;
                            sum_z += v.z;
                        }
                    }

                    grid_next[y * W + x] = { sum_x / 9.0f, sum_y / 9.0f, sum_z / 9.0f };
                }
            }

            // Swap
            std::swap(grid_curr, grid_next);

            // Zapis
            if (save_every > 0 && s % save_every == 0) {
                char path[256];
                sprintf(path, "%s/frame_%06d.ppm", outdir.c_str(), s);
                write_ppm(path, grid_curr, W, H);
            }
        }

        auto t1 = std::chrono::high_resolution_clock::now();
        double dt = std::chrono::duration<double>(t1 - t0).count();
        timings.push_back(dt);
        std::cout << "Done in " << dt << "s" << std::endl;
    }

    // Zapis statystyk do CSV
    double avg = 0, min_t = timings[0], max_t = timings[0];
    for(double t : timings) {
        avg += t;
        if(t < min_t) min_t = t;
        if(t > max_t) max_t = t;
    }
    avg /= timings.size();

    std::ofstream csv(csv_filename);
    csv << "Run,Time(s)\n";
    for(size_t i=0; i<timings.size(); i++) csv << (i+1) << "," << timings[i] << "\n";
    csv << "\nStats\nAverage," << avg << "\nMin," << min_t << "\nMax," << max_t << "\n";
    csv.close();

    std::cout << "Results saved to " << csv_filename << std::endl;

    return 0;
}