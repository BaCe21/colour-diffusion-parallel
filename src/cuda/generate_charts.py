import subprocess
import re
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys

# === KONFIGURACJA ===
# Nazwy Twoich plików exe
EXE_SEQ = "diffusion_seq.exe"
EXE_OMP = "main_openmp.exe"
EXE_MPI = "main_mpi.exe"
EXE_CUDA = "diffusion_cuda.exe"

# Parametry do testów
GRID_SIZES = [500, 1000, 2000]       # Rozdzielczości siatki
STEPS_LIST = [500, 1000, 2000]       # Ilości kroków
THREADS_OMP = [1, 2, 4, 8]           # Liczba wątków OpenMP
RANKS_MPI = [1, 2, 4]                # Liczba procesów MPI

# Domyślne wartości do testów "Fixed"
FIXED_GRID = 1000  # Kiedy zmieniamy kroki, siatka jest stała (1000x1000)
FIXED_STEPS = 1000 # Kiedy zmieniamy siatkę, kroki są stałe (1000)

results = []

def run_command(cmd_list):
    """Uruchamia komendę i zwraca czas wykonania (wersja naprawiona)"""
    
    # 1. Poprawka dla Windows: dodaj ".\" jeśli to plik exe
    exe = cmd_list[0]
    if exe.endswith(".exe") and not exe.startswith(".\\") and not os.path.isabs(exe):
        cmd_list[0] = ".\\" + exe

    print(f"Running: {' '.join(cmd_list)} ...", end=" ")
    
    try:
        # Uruchamiamy proces
        result = subprocess.run(cmd_list, capture_output=True, text=True)
        output = result.stdout
        
        # Jeśli proces zwrócił błąd
        if result.returncode != 0:
            print(f"ERROR (Kod błędu: {result.returncode})")
            # print(result.stderr) # Odkomentuj jeśli chcesz widzieć błędy
            return None

        # 2. Szukamy czasu (bardziej elastyczny regex)
        # Szuka: liczba kropka liczba, a potem 's' LUB 's' na końcu linii
        match = re.search(r"([0-9]+\.[0-9]+)\s*s", output)
        
        if match:
            time_sec = float(match.group(1))
            print(f"OK ({time_sec:.4f} s)")
            return time_sec
        else:
            print("ERROR (Nie znaleziono czasu w outputcie)")
            # Wypisz kawałek outputu, żebyśmy wiedzieli co poszło nie tak
            print(f"   RAW OUTPUT: {output[:100]}...") 
            return None
            
    except Exception as e:
        print(f"EXCEPTION: {e}")
        return None

def main():
    print("=== ROZPOCZYNAM AUTOMATYCZNE TESTY ===")

    # ---------------------------------------------------------
    # 1. OpenMP: Wpływ parametrów (Threads)
    # ---------------------------------------------------------
    
    # A) Zmienne kroki (Fixed Grid)
    for t in THREADS_OMP:
        for s in STEPS_LIST:
            cmd = [EXE_OMP, "--w", str(FIXED_GRID), "--h", str(FIXED_GRID), "--steps", str(s), "--threads", str(t), "--save_every", "0", "--repeat", "1"]
            t_sec = run_command(cmd)
            results.append({"Tech": "OpenMP", "Threads": t, "Grid": FIXED_GRID, "Steps": s, "Time": t_sec, "Type": "Steps_Scaling"})

    # B) Zmienna siatka (Fixed Steps)
    for t in THREADS_OMP:
        for g in GRID_SIZES:
            cmd = [EXE_OMP, "--w", str(g), "--h", str(g), "--steps", str(FIXED_STEPS), "--threads", str(t), "--save_every", "0", "--repeat", "1"]
            t_sec = run_command(cmd)
            results.append({"Tech": "OpenMP", "Threads": t, "Grid": g, "Steps": FIXED_STEPS, "Time": t_sec, "Type": "Grid_Scaling"})

    # ---------------------------------------------------------
    # 2. MPI: Wpływ parametrów (Ranks)
    # ---------------------------------------------------------

    # A) Zmienne kroki (Fixed Grid)
    for r in RANKS_MPI:
        for s in STEPS_LIST:
            # mpiexec -n <ranks> main_mpi.exe ...
            cmd = ["mpiexec", "-n", str(r), EXE_MPI, "--w", str(FIXED_GRID), "--h", str(FIXED_GRID), "--steps", str(s), "--save_every", "0", "--repeat", "1"]
            t_sec = run_command(cmd)
            results.append({"Tech": "MPI", "Threads": r, "Grid": FIXED_GRID, "Steps": s, "Time": t_sec, "Type": "Steps_Scaling"})

    # B) Zmienna siatka (Fixed Steps)
    for r in RANKS_MPI:
        for g in GRID_SIZES:
            cmd = ["mpiexec", "-n", str(r), EXE_MPI, "--w", str(g), "--h", str(g), "--steps", str(FIXED_STEPS), "--save_every", "0", "--repeat", "1"]
            t_sec = run_command(cmd)
            results.append({"Tech": "MPI", "Threads": r, "Grid": g, "Steps": FIXED_STEPS, "Time": t_sec, "Type": "Grid_Scaling"})

    # ---------------------------------------------------------
    # 3. Sekwencyjny (Baseline)
    # ---------------------------------------------------------
    
    # A) Zmienne kroki
    for s in STEPS_LIST:
        cmd = [EXE_SEQ, "--w", str(FIXED_GRID), "--h", str(FIXED_GRID), "--steps", str(s), "--save_every", "0", "--repeat", "1"]
        t_sec = run_command(cmd)
        results.append({"Tech": "Sequential", "Threads": 1, "Grid": FIXED_GRID, "Steps": s, "Time": t_sec, "Type": "Steps_Scaling"})

    # B) Zmienna siatka
    for g in GRID_SIZES:
        # Uwaga: dla duzej siatki seq jest bardzo wolny, mozesz tu zmniejszyc steps jesli test trwa za dlugo
        cmd = [EXE_SEQ, "--w", str(g), "--h", str(g), "--steps", str(FIXED_STEPS), "--save_every", "0", "--repeat", "1"]
        t_sec = run_command(cmd)
        results.append({"Tech": "Sequential", "Threads": 1, "Grid": g, "Steps": FIXED_STEPS, "Time": t_sec, "Type": "Grid_Scaling"})

    # ---------------------------------------------------------
    # 4. CUDA (Do porównania)
    # ---------------------------------------------------------
    
    # A) Zmienne kroki
    for s in STEPS_LIST:
        cmd = [EXE_CUDA, "--w", str(FIXED_GRID), "--h", str(FIXED_GRID), "--steps", str(s), "--tile", "32", "--save_every", "0", "--repeat", "1"]
        t_sec = run_command(cmd)
        results.append({"Tech": "CUDA", "Threads": "GPU", "Grid": FIXED_GRID, "Steps": s, "Time": t_sec, "Type": "Steps_Scaling"})

    # B) Zmienna siatka
    for g in GRID_SIZES:
        cmd = [EXE_CUDA, "--w", str(g), "--h", str(g), "--steps", str(FIXED_STEPS), "--tile", "32", "--save_every", "0", "--repeat", "1"]
        t_sec = run_command(cmd)
        results.append({"Tech": "CUDA", "Threads": "GPU", "Grid": g, "Steps": FIXED_STEPS, "Time": t_sec, "Type": "Grid_Scaling"})


    # =========================================================
    # GENEROWANIE WYKRESÓW
    # =========================================================
    print("\nGenerowanie wykresów...")
    df = pd.DataFrame(results)
    
    # Upewnij sie ze folder istnieje
    if not os.path.exists("raport_wykresy"):
        os.makedirs("raport_wykresy")

    # --- Wykres 1: OMP Scaling (Steps) ---
    plt.figure(figsize=(8,6))
    subset = df[(df["Tech"] == "OpenMP") & (df["Type"] == "Steps_Scaling")]
    for t in subset["Threads"].unique():
        data = subset[subset["Threads"] == t]
        plt.plot(data["Steps"], data["Time"], marker='o', label=f"{t} threads")
    plt.title(f"OpenMP: Czas vs Kroki (Siatka {FIXED_GRID}x{FIXED_GRID})")
    plt.xlabel("Liczba kroków")
    plt.ylabel("Czas [s]")
    plt.legend()
    plt.grid(True)
    plt.savefig("raport_wykresy/1_omp_steps.png")

    # --- Wykres 2: MPI Scaling (Steps) ---
    plt.figure(figsize=(8,6))
    subset = df[(df["Tech"] == "MPI") & (df["Type"] == "Steps_Scaling")]
    for t in subset["Threads"].unique():
        data = subset[subset["Threads"] == t]
        plt.plot(data["Steps"], data["Time"], marker='o', label=f"{t} ranks")
    plt.title(f"MPI: Czas vs Kroki (Siatka {FIXED_GRID}x{FIXED_GRID})")
    plt.xlabel("Liczba kroków")
    plt.ylabel("Czas [s]")
    plt.legend()
    plt.grid(True)
    plt.savefig("raport_wykresy/2_mpi_steps.png")

    # --- Wykres 3: OMP Scaling (Grid) ---
    plt.figure(figsize=(8,6))
    subset = df[(df["Tech"] == "OpenMP") & (df["Type"] == "Grid_Scaling")]
    for t in subset["Threads"].unique():
        data = subset[subset["Threads"] == t]
        plt.plot(data["Grid"], data["Time"], marker='o', label=f"{t} threads")
    plt.title(f"OpenMP: Czas vs Rozmiar Siatki (Kroki: {FIXED_STEPS})")
    plt.xlabel("Rozmiar boku siatki")
    plt.ylabel("Czas [s]")
    plt.legend()
    plt.grid(True)
    plt.savefig("raport_wykresy/3_omp_grid.png")

    # --- Wykres 4: MPI Scaling (Grid) ---
    plt.figure(figsize=(8,6))
    subset = df[(df["Tech"] == "MPI") & (df["Type"] == "Grid_Scaling")]
    for t in subset["Threads"].unique():
        data = subset[subset["Threads"] == t]
        plt.plot(data["Grid"], data["Time"], marker='o', label=f"{t} ranks")
    plt.title(f"MPI: Czas vs Rozmiar Siatki (Kroki: {FIXED_STEPS})")
    plt.xlabel("Rozmiar boku siatki")
    plt.ylabel("Czas [s]")
    plt.legend()
    plt.grid(True)
    plt.savefig("raport_wykresy/4_mpi_grid.png")

    # --- Wykres 5: Porównanie technologii (Steps) ---
    # Bierzemy najlepsze OMP (8), najlepsze MPI (4), Seq i CUDA
    plt.figure(figsize=(8,6))
    
    # Seq
    seq_data = df[(df["Tech"] == "Sequential") & (df["Type"] == "Steps_Scaling")]
    plt.plot(seq_data["Steps"], seq_data["Time"], marker='x', linestyle='--', label="Sequential")
    
    # Best OMP (max threads)
    best_omp = df[(df["Tech"] == "OpenMP") & (df["Type"] == "Steps_Scaling") & (df["Threads"] == max(THREADS_OMP))]
    plt.plot(best_omp["Steps"], best_omp["Time"], marker='o', label="OpenMP (Best)")

    # Best MPI (max ranks)
    best_mpi = df[(df["Tech"] == "MPI") & (df["Type"] == "Steps_Scaling") & (df["Threads"] == max(RANKS_MPI))]
    plt.plot(best_mpi["Steps"], best_mpi["Time"], marker='s', label="MPI (Best)")

    # CUDA
    cuda_data = df[(df["Tech"] == "CUDA") & (df["Type"] == "Steps_Scaling")]
    plt.plot(cuda_data["Steps"], cuda_data["Time"], marker='^', linewidth=2, label="CUDA")

    plt.title("Porównanie wydajności: Zmienna liczba kroków")
    plt.xlabel("Kroki")
    plt.ylabel("Czas [s]")
    plt.legend()
    plt.grid(True)
    plt.savefig("raport_wykresy/5_comparison_steps.png")

    # --- Wykres 6: Porównanie technologii (Grid) ---
    plt.figure(figsize=(8,6))
    
    seq_data = df[(df["Tech"] == "Sequential") & (df["Type"] == "Grid_Scaling")]
    plt.plot(seq_data["Grid"], seq_data["Time"], marker='x', linestyle='--', label="Sequential")
    
    best_omp = df[(df["Tech"] == "OpenMP") & (df["Type"] == "Grid_Scaling") & (df["Threads"] == max(THREADS_OMP))]
    plt.plot(best_omp["Grid"], best_omp["Time"], marker='o', label="OpenMP (Best)")

    best_mpi = df[(df["Tech"] == "MPI") & (df["Type"] == "Grid_Scaling") & (df["Threads"] == max(RANKS_MPI))]
    plt.plot(best_mpi["Grid"], best_mpi["Time"], marker='s', label="MPI (Best)")

    cuda_data = df[(df["Tech"] == "CUDA") & (df["Type"] == "Grid_Scaling")]
    plt.plot(cuda_data["Grid"], cuda_data["Time"], marker='^', linewidth=2, label="CUDA")

    plt.title("Porównanie wydajności: Zmienny rozmiar siatki")
    plt.xlabel("Rozmiar siatki")
    plt.ylabel("Czas [s]")
    plt.legend()
    plt.grid(True)
    plt.savefig("raport_wykresy/6_comparison_grid.png")

    # --- Wykres 7: Seq (Grid) ---
    plt.figure(figsize=(8,6))
    subset = df[(df["Tech"] == "Sequential") & (df["Type"] == "Grid_Scaling")]
    plt.bar(subset["Grid"].astype(str), subset["Time"], color='gray')
    plt.title("Czas sekwencyjny vs Rozmiar siatki")
    plt.ylabel("Czas [s]")
    plt.savefig("raport_wykresy/7_seq_grid.png")

    # --- Wykres 8: Seq (Steps) ---
    plt.figure(figsize=(8,6))
    subset = df[(df["Tech"] == "Sequential") & (df["Type"] == "Steps_Scaling")]
    plt.bar(subset["Steps"].astype(str), subset["Time"], color='gray')
    plt.title("Czas sekwencyjny vs Liczba kroków")
    plt.ylabel("Czas [s]")
    plt.savefig("raport_wykresy/8_seq_steps.png")

    print("\nGotowe! Wykresy zapisano w folderze 'raport_wykresy'.")

if __name__ == "__main__":
    main()