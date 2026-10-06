#include <mpi.h>
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <algorithm>
#include <numeric>

namespace fs = std::filesystem;

struct float3f { float x, y, z; };

// Funkcje pomocnicze 
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
    int& W, int& H, int& steps, int& save_every, std::string& outdir, int& repeat)
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



// Inicjalizacja źródeł 
void init_local_sources(std::vector<float3f>& local_grid, int W, int H, 
                       int start_row, int local_rows, int halo_rows) {
    
    std::fill(local_grid.begin(), local_grid.end(), float3f{0.f, 0.f, 0.f});

    const float beta = 4.0f;
    const float intensity = 1.8f;

    auto put = [&](int cx, int cy, float r, float g, float b, int radius) {
        const int r2 = radius * radius;
        int min_y = std::max(cy - radius, start_row - 1); 
        int max_y = std::min(cy + radius, start_row + local_rows); 

        for (int gy = min_y; gy <= max_y; ++gy) {
            int ly = gy - start_row + 1; 
            
            if (ly < 0 || ly >= local_rows + 2) continue;

            for (int gx = cx - radius; gx <= cx + radius; ++gx) {
                if (gx < 0 || gx >= W) continue;
                
                int dx = gx - cx;
                int dy = gy - cy;
                int d2 = dx * dx + dy * dy;
                
                if (d2 > r2) continue;

                float dist = sqrtf((float)d2);
                float t = dist / (float)radius;
                float w = 1.0f + (beta - 1.0f) * t;
                float f = w * intensity;

                int idx = ly * W + gx;
                local_grid[idx].x += r * f;
                local_grid[idx].y += g * f;
                local_grid[idx].z += b * f;
            }
        }
    };

    put(W / 4, H / 3, 1.f, 0.f, 0.f, 150);
    put(3 * W / 4, 2 * H / 3, 0.f, 1.f, 0.f, 150);
    put(W / 2, H / 2, 0.f, 0.f, 1.f, 150);
}


int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int W = 1000, H = 1000, steps = 100000;
    int save_every = 500;
    int repeat = 1;
    std::string outdir = "frames_mpi";

    parse_args(argc, argv, W, H, steps, save_every, outdir, repeat);

    if (rank == 0) {
        if (!fs::exists(outdir)) fs::create_directories(outdir);
        std::string csv_filename = "diffusion_mpi_" + std::to_string(size) + "ranks_" +
                                   std::to_string(W) + "x" + std::to_string(H) + ".csv";
        std::cout << "MPI Run: " << size << " ranks, Grid: " << W << "x" << H << std::endl;
        std::cout << "Logging to: " << csv_filename << std::endl;
    }

    int base_rows = H / size;
    int remainder = H % size;
    int my_rows = base_rows + (rank < remainder ? 1 : 0);
    
    int my_start_row = 0;
    for (int r = 0; r < rank; ++r) {
        my_start_row += base_rows + (r < remainder ? 1 : 0);
    }

    int halo_rows = 2; 
    int buffer_height = my_rows + halo_rows;
    size_t buffer_size = (size_t)buffer_height * W;

    std::vector<float3f> grid_curr(buffer_size);
    std::vector<float3f> grid_next(buffer_size);

    std::vector<int> recvcounts(size);
    std::vector<int> displs(size);
    if (rank == 0) {
        int offset = 0;
        for (int r = 0; r < size; ++r) {
            int rows = base_rows + (r < remainder ? 1 : 0);
            recvcounts[r] = rows * W * 3; 
            displs[r] = offset;
            offset += recvcounts[r];
        }
    }

    std::vector<double> timings;

    // Pętla powtórzeń 
    for (int run = 0; run < repeat; ++run) {
        if(rank == 0) std::cout << "Run " << (run+1) << "/" << repeat << "... " << std::flush;
        
        init_local_sources(grid_curr, W, H, my_start_row, my_rows, halo_rows);
        grid_next = grid_curr;

        MPI_Barrier(MPI_COMM_WORLD);
        double t0 = MPI_Wtime();

        for (int s = 0; s < steps; ++s) {
            int top_neighbor = (rank == 0) ? MPI_PROC_NULL : rank - 1;
            int bot_neighbor = (rank == size - 1) ? MPI_PROC_NULL : rank + 1;

            MPI_Sendrecv(
                &grid_curr[1 * W], W * 3, MPI_FLOAT, top_neighbor, 0,
                &grid_curr[(my_rows + 1) * W], W * 3, MPI_FLOAT, bot_neighbor, 0,
                MPI_COMM_WORLD, MPI_STATUS_IGNORE
            );
                    if (rank == 0)
                    {
                std::copy(
                    grid_curr.begin() + W,
                    grid_curr.begin() + 2 * W,
                    grid_curr.begin()
                        );
                    }

                    if (rank == size - 1)
                    {
                std::copy(
                    grid_curr.begin() + my_rows * W,
                    grid_curr.begin() + (my_rows + 1) * W,
                    grid_curr.begin() + (my_rows + 1) * W
                        );
                    }
            MPI_Sendrecv(
                &grid_curr[my_rows * W], W * 3, MPI_FLOAT, bot_neighbor, 1,
                &grid_curr[0 * W], W * 3, MPI_FLOAT, top_neighbor, 1,
                MPI_COMM_WORLD, MPI_STATUS_IGNORE
            );
                if (rank == 0)
                {
                    for (int x = 0; x < W; ++x)
                    {
                        grid_curr[x] =
                            grid_curr[W + x];
                    }
                }

                if (rank == size - 1)
                {
                    for (int x = 0; x < W; ++x)
                    {
                        grid_curr[
                            (my_rows + 1) * W + x
                        ] =
                            grid_curr[
                                my_rows * W + x
                            ];
                    }
                }
            // 2. Obliczenia
            for (int y = 1; y <= my_rows; ++y) {
                for (int x = 0; x < W; ++x) {
                    float sum_x = 0, sum_y = 0, sum_z = 0;
                    
                    // Pętla 3x3
                    for (int dy = -1; dy <= 1; ++dy) {
                        for (int dx = -1; dx <= 1; ++dx) {
                           
                            int nx = std::max(0, std::min(W - 1, x + dx));
                            int ny = y + dy;
                            
                            float3f v = grid_curr[ny * W + nx];
                            sum_x += v.x;
                            sum_y += v.y;
                            sum_z += v.z;
                        }
                    }
                    grid_next[y * W + x] = { sum_x / 9.0f, sum_y / 9.0f, sum_z / 9.0f };
                }
            }

            std::swap(grid_curr, grid_next);

            // 3. Zapisywanie klatki
            if (save_every > 0 && s % save_every == 0) {

                std::vector<float3f> full_grid;
                if (rank == 0) full_grid.resize(W * H);

                MPI_Gatherv(
                    &grid_curr[1 * W], my_rows * W * 3, MPI_FLOAT,
                    full_grid.data(), recvcounts.data(), displs.data(), MPI_FLOAT,
                    0, MPI_COMM_WORLD
                );

                if (rank == 0) {
                    char path[256];
                    sprintf(path, "%s/frame_%06d.ppm", outdir.c_str(), s);
                    write_ppm(path, full_grid, W, H);
                }
            }
        }

        MPI_Barrier(MPI_COMM_WORLD);
        double t1 = MPI_Wtime();
        timings.push_back(t1 - t0);
        if(rank == 0) std::cout << "Done in " << (t1 - t0) << "s" << std::endl;
    }

    // Zapis wyników do CSV
    if (rank == 0) {
        double avg = 0, min_t = timings[0], max_t = timings[0];
        for(double t : timings) {
            avg += t;
            if(t < min_t) min_t = t;
            if(t > max_t) max_t = t;
        }
        avg /= timings.size();

        std::string csv_filename = "diffusion_mpi_" + std::to_string(size) + "ranks_" +
                                   std::to_string(W) + "x" + std::to_string(H) + "_" + 
                                   std::to_string(steps) + ".csv";
        
        std::ofstream csv(csv_filename);
        csv << "Run,Time(s)\n";
        for(size_t i=0; i<timings.size(); i++) csv << (i+1) << "," << timings[i] << "\n";
        csv << "\nStats\nAverage," << avg << "\nMin," << min_t << "\nMax," << max_t << "\n";
        csv.close();
    }

    MPI_Finalize();
    return 0;
}
