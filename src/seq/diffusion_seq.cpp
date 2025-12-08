#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <string>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <numeric>
#include <iostream>
#include <cmath>

struct float3f { float x, y, z; };

float3f avg_stencil_9(const float3f* data, int width, int height, int x, int y) {
    float3f sum = { 0.f, 0.f, 0.f };
    int count = 0;

    for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
            int nx = x + dx;
            int ny = y + dy;

            if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                float3f v = data[ny * width + nx];
                sum.x += v.x;
                sum.y += v.y;
                sum.z += v.z;
                count++;
            }
        }
    }

    if (count > 0) {
        sum.x /= count;
        sum.y /= count;
        sum.z /= count;
    }

    return sum;
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

void clear_directory(const std::string& outdir) {
    if (std::filesystem::exists(outdir)) {
        for (const auto& entry : std::filesystem::directory_iterator(outdir)) {
            std::filesystem::remove_all(entry.path());
        }
        std::cout << "Cleared directory: " << outdir << std::endl;
    }
}

void parse_args(int argc, char** argv,
    int& W, int& H, int& steps,
    int& save_every, std::string& outdir, int& repeat)
{
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--w") && i + 1 < argc) W = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--h") && i + 1 < argc) H = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--steps") && i + 1 < argc) steps = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--save_every") && i + 1 < argc) save_every = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--outdir") && i + 1 < argc) outdir = argv[++i];
        else if (!strcmp(argv[i], "--repeat") && i + 1 < argc) repeat = atoi(argv[++i]);
    }
}

void diffuse_step(const std::vector<float3f>& in, std::vector<float3f>& out, int W, int H) {
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            out[y * W + x] = avg_stencil_9(in.data(), W, H, x, y);
        }
    }
}

int main(int argc, char** argv) {
    int W = 1000, H = 1000, steps = 100000;
    int save_every = 500;
    int repeat = 1;
    std::string outdir = "frames_seq";

    parse_args(argc, argv, W, H, steps, save_every, outdir, repeat);

    // Create CSV file name
    std::string csv_filename = "diffusion_seq_" + std::to_string(W) +
        "x" + std::to_string(H) + "_" +
        std::to_string(steps) + ".csv";

    // Vector to store timing results
    std::vector<double> timings;

    // Run simulations
    for (int run = 0; run < repeat; ++run) {
        std::cout << "\n=== SEQ Simulation Run " << (run + 1) << "/" << repeat << " ===" << std::endl;

        // Clear output directory before each simulation
        clear_directory(outdir);
        std::filesystem::create_directories(outdir);

        size_t N = (size_t)W * H;
        std::vector<float3f> buf_a(N);
        std::vector<float3f> buf_b(N);

        place_sources(buf_a, W, H);

        auto t0 = std::chrono::high_resolution_clock::now();
        for (int s = 0; s < steps; ++s) {
            // Perform diffusion step
            diffuse_step(buf_a, buf_b, W, H);

            // Swap buffers
            std::swap(buf_a, buf_b);

            if ((s % save_every) == 0 && save_every > 0) {
                char path[1024];
                sprintf(path, "%s/frame_%06d.ppm", outdir.c_str(), s);
                write_ppm(path, buf_a, W, H);

                double total = 0.0;
                for (size_t i = 0; i < N; ++i) {
                    total += buf_a[i].x + buf_a[i].y + buf_a[i].z;
                }
                printf("step %d total energy: %.6f\n", s, total);
            }
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        double secs = std::chrono::duration<double>(t1 - t0).count();

        std::cout << "Done " << steps << " steps on " << W << "x" << H
            << " in " << secs << " s (Run " << (run + 1) << ")" << std::endl;

        // Store timing for this run
        timings.push_back(secs);

        // Save final frame
        write_ppm(outdir + "/frame_final.ppm", buf_a, W, H);
    }

    // Calculate statistics
    double avg_time = 0.0;
    double min_time = timings[0];
    double max_time = timings[0];

    for (double t : timings) {
        avg_time += t;
        if (t < min_time) min_time = t;
        if (t > max_time) max_time = t;
    }
    avg_time /= timings.size();

    // Write results to CSV
    std::ofstream csv_file(csv_filename);
    if (csv_file.is_open()) {
        csv_file << "Run,Time(s)\n";
        for (size_t i = 0; i < timings.size(); ++i) {
            csv_file << (i + 1) << "," << timings[i] << "\n";
        }
        csv_file << "\nStatistics\n";
        csv_file << "Average," << avg_time << "\n";
        csv_file << "Min," << min_time << "\n";
        csv_file << "Max," << max_time << "\n";
        csv_file << "Runs," << timings.size() << "\n";
        csv_file.close();

        std::cout << "\n=== SEQ Results Summary ===" << std::endl;
        std::cout << "Timing data saved to: " << csv_filename << std::endl;
        std::cout << "Average time: " << avg_time << " s" << std::endl;
        std::cout << "Minimum time: " << min_time << " s" << std::endl;
        std::cout << "Maximum time: " << max_time << " s" << std::endl;
        std::cout << "Number of runs: " << timings.size() << std::endl;

        // Calculate performance metrics
        double total_pixels = (double)W * H * steps;
        double mpps = total_pixels / (avg_time * 1e6);  // Million pixels per second
        std::cout << "Performance: " << mpps << " MPixels/s" << std::endl;
    }
    else {
        std::cerr << "Error: Could not open CSV file for writing: " << csv_filename << std::endl;
    }

    return 0;
}