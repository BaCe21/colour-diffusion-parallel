import os
import glob
import re
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# ==========================================
# KONFIGURACJA
# ==========================================

# Wybierz folder, z którego chcesz skleić GIFa:
# "frames_omp" -> wyniki z OpenMP
# "frames_mpi" -> wyniki z MPI
# "frames"     -> wyniki z CUDA (oryginalne)
SOURCE_DIR = "frames_omp" 

OUTPUT_FILE = "simulation_result.gif"
FPS = 20  # Prędkość animacji (klatki na sekundę)

# ==========================================

def natural_sort_key(s):
    """Sortowanie naturalne (żeby frame_10 był po frame_9, a nie po frame_1)"""
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split('([0-9]+)', s)]

def create_gif():
    # 1. Sprawdzenie katalogu
    if not os.path.exists(SOURCE_DIR):
        print(f"BŁĄD: Katalog '{SOURCE_DIR}' nie istnieje!")
        print("Upewnij się, że uruchomiłeś program C++ i wygenerował on klatki.")
        return

    # 2. Szukanie plików .ppm
    search_path = os.path.join(SOURCE_DIR, "*.ppm")
    files = sorted(glob.glob(search_path), key=natural_sort_key)
    
    if not files:
        print(f"BŁĄD: W katalogu '{SOURCE_DIR}' nie ma żadnych plików .ppm.")
        return

    print(f"Znaleziono {len(files)} klatek w '{SOURCE_DIR}'.")
    print("Wczytywanie obrazów do pamięci (to może chwilę potrwać)...")

    # 3. Wczytywanie obrazów
    images = []
    for filename in files:
        try:
            # Otwieramy i kopiujemy obraz, zamykając plik
            with Image.open(filename) as img:
                images.append(img.copy())
        except Exception as e:
            print(f"Ostrzeżenie: Nie udało się wczytać {filename}: {e}")

    if not images:
        print("Brak poprawnych obrazów.")
        return

    # 4. Tworzenie animacji
    print(f"Generowanie pliku GIF: {OUTPUT_FILE}...")
    
    fig, ax = plt.subplots(figsize=(8, 8))
    plt.axis('off') # Ukryj osie
    
    # Inicjalizacja pierwszej klatki
    im = ax.imshow(images[0], animated=True)
    
    def update(frame_img):
        im.set_data(frame_img)
        return [im]

    # Tworzenie obiektu animacji
    ani = animation.FuncAnimation(
        fig, 
        update, 
        frames=images, 
        interval=1000/FPS, # interval w ms
        blit=True
    )

    # Zapis
    try:
        ani.save(OUTPUT_FILE, writer='pillow', fps=FPS)
        print(f"SUKCES! Zapisano animację jako: {os.path.abspath(OUTPUT_FILE)}")
    except Exception as e:
        print(f"Błąd podczas zapisywania GIF: {e}")
        
    plt.close() # Zamknij okno wykresu w tle

if __name__ == "__main__":
    create_gif()