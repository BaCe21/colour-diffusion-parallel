// src/mpi/main_mpi.cpp
// MPI implementation of 2D color diffusion (row-wise domain decomposition).

#include <mpi.h>
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <cmath>
#include <cstring>
#include <filesystem>   // C++17
namespace fs = std::filesystem;

struct Pixel { float r, g, b; };
static inline void clamp01(float &x) { if (x < 0.f) x = 0.f; if (x > 1.f) x = 1.f; }

void save_ppm(const std::string &filename, const std::vector<Pixel>& grid, int w, int h) {
    std::ofstream f(filename, std::ios::binary);
    f << "P6\n" << w << " " << h << "\n255\n";
    for (int i = 0; i < w*h; ++i) {
        unsigned char r = (unsigned char)std::lround(std::max(0.f,std::min(1.f,grid[i].r))*255.0f);
        unsigned char g = (unsigned char)std::lround(std::max(0.f,std::min(1.f,grid[i].g))*255.0f);
        unsigned char b = (unsigned char)std::lround(std::max(0.f,std::min(1.f,grid[i].b))*255.0f);
        f.put((char)r); f.put((char)g); f.put((char)b);
    }
    f.close();
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int N = 1000, iters = 500, save_interval = 0, stencil = 5;

    for (int i=1;i<argc;i++){
        if (strcmp(argv[i], "--size")==0 && i+1<argc) N = atoi(argv[++i]);
        else if (strcmp(argv[i], "--iters")==0 && i+1<argc) iters = atoi(argv[++i]);
        else if (strcmp(argv[i], "--save_interval")==0 && i+1<argc) save_interval = atoi(argv[++i]);
        else if (strcmp(argv[i], "--stencil")==0 && i+1<argc) stencil = atoi(argv[++i]);
        else if (strcmp(argv[i], "--help")==0) {
            if(rank==0) std::cout << "Usage: --size N --iters K --save_interval M --stencil 5|9\n";
            MPI_Finalize(); return 0;
        }
    }

    if(rank==0) {
        std::cout << "MPI size: " << size << ", Grid: " << N << "x" << N
                  << ", iters=" << iters << ", stencil=" << stencil << "\n";
        if(save_interval>0 && !fs::exists("frames")) fs::create_directory("frames");
    }

    // row decomposition
    int base = N / size;
    int rem = N % size;
    int local_rows = base + (rank<rem ? 1:0);
    int buf_rows = local_rows + 2; // halo
    int cols = N;

    // offsets for gathering
    std::vector<int> recvcounts(size), displs(size);
    for(int r=0, offset=0;r<size;++r){
        int rows_r = base + (r<rem?1:0);
        recvcounts[r] = rows_r * cols;
        displs[r] = offset;
        offset += recvcounts[r];
    }

    // allocate grids
    std::vector<Pixel> grid(buf_rows*cols), newgrid(buf_rows*cols);
    auto idx = [&](int y,int x){ return y*cols + x; };

    // helper to set pixel on global grid
    auto set_global_px = [&](int gx,int gy,float r,float g,float b){
        if(gx<0||gx>=N||gy<0||gy>=N) return;
        int start_row = 0;
        for(int rr=0;rr<rank;++rr) start_row += base + (rr<rem?1:0);
        int local_y = gy - start_row;
        if(local_y>=0 && local_y<local_rows){
            grid[idx(local_y+1,gx)] = {r,g,b};
        }
    };

    // initial sources
    std::vector<std::tuple<int,int,float,float,float>> sources = {
        {N/2,N/2,1.f,0.f,0.f},
        {N/4,N/4,0.f,1.f,0.f},
        {3*N/4,3*N/4,0.f,0.f,1.f}
    };
    int radius = std::max(1,N/100);
    for(auto [gx,gy,r,g,b]:sources){
        for(int dy=-radius;dy<=radius;++dy)
        for(int dx=-radius;dx<=radius;++dx)
            if(dx*dx+dy*dy<=radius*radius) set_global_px(gx+dx,gy+dy,r,g,b);
    }

    // MPI datatype
    MPI_Datatype MPI_PIXEL;
    MPI_Type_contiguous(3,MPI_FLOAT,&MPI_PIXEL);
    MPI_Type_commit(&MPI_PIXEL);

    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();

    for(int it=0;it<iters;++it){
        // restore sources each iteration
        for(auto [gx,gy,r,g,b]:sources){
            for(int dy=-radius;dy<=radius;++dy)
            for(int dx=-radius;dx<=radius;++dx)
                if(dx*dx+dy*dy<=radius*radius) set_global_px(gx+dx,gy+dy,r,g,b);
        }

        // halo exchange
        int top = rank-1, bottom=rank+1;
        if(top<0) top=MPI_PROC_NULL;
        if(bottom>=size) bottom=MPI_PROC_NULL;

        MPI_Sendrecv(&grid[idx(1,0)], cols, MPI_PIXEL, top,0,
                     &grid[idx(0,0)], cols, MPI_PIXEL, top,1,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
        MPI_Sendrecv(&grid[idx(local_rows,0)], cols, MPI_PIXEL, bottom,1,
                     &grid[idx(local_rows+1,0)], cols, MPI_PIXEL, bottom,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);

        // update
        for(int y=1;y<=local_rows;++y){
            for(int x=0;x<cols;++x){
                Pixel acc={0.f,0.f,0.f}; float count=0.f;
                for(int dy=-1;dy<=1;++dy)
                    for(int dx=-1;dx<=1;++dx){
                        if(stencil==5 && abs(dy)+abs(dx)>1) continue;
                        int yy=y+dy, xx=x+dx;
                        if(yy>=0 && yy<buf_rows && xx>=0 && xx<cols){
                            acc.r += grid[idx(yy,xx)].r;
                            acc.g += grid[idx(yy,xx)].g;
                            acc.b += grid[idx(yy,xx)].b;
                            count += 1.f;
                        }
                    }
                newgrid[idx(y,x)] = {acc.r/count,acc.g/count,acc.b/count};
            }
        }

        grid.swap(newgrid);

        // save frame
        if(save_interval>0 && ((it+1)%save_interval==0)){
            std::ostringstream ss;
            ss << "frames/frame_" << std::setfill('0') << std::setw(5) << (it+1) << ".ppm";
            // gather full grid to rank 0
            std::vector<Pixel> local_out(local_rows*cols);
            for(int y=0;y<local_rows;++y) std::memcpy(&local_out[y*cols],&grid[idx(y+1,0)],sizeof(Pixel)*cols);

            std::vector<Pixel> full;
            if(rank==0) full.resize(N*N);

            MPI_Gatherv(local_out.data(), local_rows*cols, MPI_PIXEL,
                        full.data(), recvcounts.data(), displs.data(), MPI_PIXEL,
                        0,MPI_COMM_WORLD);

            if(rank==0) save_ppm(ss.str(), full, N, N);
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double t1 = MPI_Wtime();
    double elapsed = t1-t0;

    // gather final grid
    std::vector<Pixel> local_out(local_rows*cols);
    for(int y=0;y<local_rows;++y) std::memcpy(&local_out[y*cols],&grid[idx(y+1,0)],sizeof(Pixel)*cols);

    std::vector<Pixel> full;
    if(rank==0) full.resize(N*N);

    MPI_Gatherv(local_out.data(), local_rows*cols, MPI_PIXEL,
                full.data(), recvcounts.data(), displs.data(), MPI_PIXEL,
                0,MPI_COMM_WORLD);

    if(rank==0){
        double cells = double(N)*double(N)*double(iters);
        std::cout << "MPI Done. Time: " << elapsed << " s, Throughput: " << (cells/elapsed)/1e6 << " Mcells/s\n";
        save_ppm("final_mpi.ppm", full, N, N);
        std::cout << "Wrote final_mpi.ppm\n";
    }

    MPI_Type_free(&MPI_PIXEL);
    MPI_Finalize();
    return 0;
}
