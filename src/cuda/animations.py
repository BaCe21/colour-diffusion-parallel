import os
import glob
import re
from PIL import Image

# === KONFIGURACJA ===
# Słownik mapujący nazwę trybu na folder z klatkami
DIRS = {
    "CUDA": "frames",
    "MPI":  "frames_mpi",
    "OpenMP": "frames_omp",
    "Sequential": "frames_seq"
}

# Parametry GIFa
DURATION = 50   # Czas trwania jednej klatki w ms (50ms = 20 FPS)
LOOP = 0        # 0 = nieskończona pętla

def natural_sort_key(s):
    """Sortowanie naturalne, np. frame_2.ppm przed frame_10.ppm"""
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split('([0-9]+)', s)]

def create_gif(mode_name, folder_path):
    output_filename = f"animation_{mode_name.lower()}.gif"
    
    print(f"\n--- Przetwarzanie: {mode_name} ({folder_path}) ---")

    if not os.path.exists(folder_path):
        print(f"POMINIĘTO: Folder '{folder_path}' nie istnieje (nie uruchamiałeś tej wersji?).")
        return

    # Szukanie plików .ppm
    search_pattern = os.path.join(folder_path, "*.ppm")
    files = sorted(glob.glob(search_pattern), key=natural_sort_key)

    if not files:
        print(f"POMINIĘTO: Folder '{folder_path}' jest pusty.")
        return

    print(f"Znaleziono {len(files)} klatek. Wczytywanie...")

    images = []
    try:
        # Wczytujemy obrazy
        for i, filepath in enumerate(files):
            # Opcjonalnie: prosty pasek postępu co 100 klatek
            if i % 100 == 0:
                print(f"   Wczytano {i}/{len(files)}...", end="\r")
            
            with Image.open(filepath) as img:
                images.append(img.copy())
        
        print(f"   Wczytano {len(files)}/{len(files)}. Generowanie GIF...")

        # Zapisywanie GIFa
        # save_all=True -> zapisz jako animację
        # append_images -> lista pozostałych klatek
        # duration -> czas klatki w ms
        # loop -> 0 to pętla nieskończona
        images[0].save(
            output_filename,
            save_all=True,
            append_images=images[1:],
            optimize=False,
            duration=DURATION,
            loop=LOOP
        )
        
        print(f"SUKCES! Zapisano plik: {os.path.abspath(output_filename)}")

    except Exception as e:
        print(f"BŁĄD: Coś poszło nie tak przy tworzeniu {output_filename}: {e}")

def main():
    print("=== KREATOR ANIMACJI DYFUZJI ===")
    print("Dostępne tryby:")
    
    # Wyświetl dostępne opcje
    available_modes = []
    for key, folder in DIRS.items():
        status = "DOSTĘPNY" if os.path.exists(folder) and os.listdir(folder) else "BRAK DANYCH"
        print(f"  [{key}] -> folder: {folder} ({status})")
        if status == "DOSTĘPNY":
            available_modes.append(key)

    print("\nCo chcesz zrobić?")
    print("1. Generuj GIF tylko dla CUDA")
    print("2. Generuj GIF tylko dla OpenMP")
    print("3. Generuj GIF tylko dla MPI")
    print("4. Generuj GIF tylko dla Sekwencyjnego")
    print("5. Generuj WSZYSTKIE dostępne")
    print("0. Wyjście")

    choice = input("\nWybierz opcję (0-5): ").strip()

    if choice == "1":
        create_gif("CUDA", DIRS["CUDA"])
    elif choice == "2":
        create_gif("OpenMP", DIRS["OpenMP"])
    elif choice == "3":
        create_gif("MPI", DIRS["MPI"])
    elif choice == "4":
        create_gif("Sequential", DIRS["Sequential"])
    elif choice == "5":
        for name, folder in DIRS.items():
            create_gif(name, folder)
    elif choice == "0":
        print("Pa pa!")
    else:
        print("Nieznana opcja.")

if __name__ == "__main__":
    main()