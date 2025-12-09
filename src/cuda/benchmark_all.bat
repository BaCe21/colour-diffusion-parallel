@echo off
echo === ROZPOCZYNAM AUTOMATYCZNY BENCHMARK ===
echo Uwaga: To moze potrwac kilka minut.

set STEPS=1000
set REPEAT=3
set SAVE=0

:: --- 500x500 ---
echo [TEST 1/3] Size 500x500...
diffusion_seq.exe --w 500 --h 500 --steps %STEPS% --save_every %SAVE% --repeat %REPEAT%
main_openmp.exe   --w 500 --h 500 --steps %STEPS% --save_every %SAVE% --threads 8 --repeat %REPEAT%
call run_mpi.bat  --w 500 --h 500 --steps %STEPS% --save_every %SAVE% --repeat %REPEAT%
diffusion_cuda.exe --w 500 --h 500 --steps %STEPS% --save_every %SAVE% --repeat %REPEAT%

:: --- 1000x1000 ---
echo [TEST 2/3] Size 1000x1000...
diffusion_seq.exe --w 1000 --h 1000 --steps %STEPS% --save_every %SAVE% --repeat %REPEAT%
main_openmp.exe   --w 1000 --h 1000 --steps %STEPS% --save_every %SAVE% --threads 8 --repeat %REPEAT%
call run_mpi.bat  --w 1000 --h 1000 --steps %STEPS% --save_every %SAVE% --repeat %REPEAT%
diffusion_cuda.exe --w 1000 --h 1000 --steps %STEPS% --save_every %SAVE% --repeat %REPEAT%

:: --- 2000x2000 ---
echo [TEST 3/3] Size 2000x2000...
:: UWAGA: Sekwencyjny przy 2000x2000 bedzie trwal wieki, mozna zmniejszyc steps dla niego, ale zaklamie to wynik.
:: Dla testu puszczamy:
diffusion_seq.exe --w 2000 --h 2000 --steps 200 --save_every 0 --repeat 1
main_openmp.exe   --w 2000 --h 2000 --steps %STEPS% --save_every %SAVE% --threads 8 --repeat %REPEAT%
call run_mpi.bat  --w 2000 --h 2000 --steps %STEPS% --save_every %SAVE% --repeat %REPEAT%
diffusion_cuda.exe --w 2000 --h 2000 --steps %STEPS% --save_every %SAVE% --repeat %REPEAT%

echo === KONIEC ===
pause