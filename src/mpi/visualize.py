import os
import glob
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import math
import re

# --- KONFIGURACJA ---
# Zmień na "frames_mpi" jeśli chcesz oglądać wynik MPI
FRAME_DIR = "frames_mpi"  
OUTPUT_GIF = "flood_sim.gif"

# Funkcja do naturalnego sortowania (żeby frame_2 nie był po frame_10)
def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split('([0-9]+)', s)]

# Sprawdzenie czy katalog istnieje
if not os.path.exists(FRAME_DIR):
    print(f"BŁĄD: Katalog '{FRAME_DIR}' nie istnieje! Uruchom najpierw symulację.")
    exit()

# Wczytywanie ścieżek
frame_paths = sorted(glob.glob(os.path.join(FRAME_DIR, "*.ppm")), key=natural_sort_key)
num_frames = len(frame_paths)

if num_frames == 0:
    print(f"BŁĄD: Nie znaleziono plików .ppm w katalogu '{FRAME_DIR}'")
    exit()

print(f"Znaleziono {num_frames} klatek w '{FRAME_DIR}'. Wczytywanie...")

# Wczytywanie obrazów do pamięci
frames = []
for fp in frame_paths:
    # Otwieramy i kopiujemy, żeby zamknąć plik
    with Image.open(fp) as img:
        frames.append(img.copy())

print("Tworzenie podglądu statycznego...")

# --- RYSOWANIE PRZEGLĄDU (Co 10%) ---
percent_frames = []
for p in range(10, 101, 10):
    # Wybieramy klatkę odpowiadającą p% postępu
    idx = math.floor((p / 100) * (num_frames - 1))
    percent_frames.append((p, frames[idx]))

cols = 5
rows = math.ceil(len(percent_frames) / cols)

plt.figure(figsize=(15, 6))
for i, (p, fr) in enumerate(percent_frames, 1):
    plt.subplot(rows, cols, i)
    plt.imshow(fr)
    plt.title(f"{p}% (Klatka {int((p/100)*(num_frames-1))})")
    plt.axis("off")

plt.tight_layout()
print("Wyświetlam podgląd. Zamknij okno, aby generować GIF...")
plt.show()

# --- GENEROWANIE GIF ---
print(f"Generowanie animacji '{OUTPUT_GIF}'... to może chwilę potrwać.")

fig, ax = plt.subplots()
plt.axis('off')
# Wyświetl pierwszą klatkę jako bazę
im = ax.imshow(frames[0], animated=True)

def update(frame_img):
    im.set_data(frame_img)
    return [im]

# interval=50 to 50ms na klatkę (czyli 20 FPS)
ani = animation.FuncAnimation(fig, update, frames=frames, interval=50, blit=True)

# Zapisywanie (używamy pillow, bo jest wbudowane w biblioteki Pythona)
ani.save(OUTPUT_GIF, writer='pillow', fps=30)

print("Gotowe! Zapisano plik GIF.")