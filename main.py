
# ============================================================
# PROJECT PCD SESI 5 - SPATIAL FILTERING
# ============================================================
# Folder Project : PCD_Sesi5_Project_Spatial_Filter
# File Program   : main.py
# Citra Input    : image.jpg
# ============================================================

import imageio.v2 as imageio
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from pathlib import Path
import csv


# ============================================================
# 1. PENGATURAN FOLDER DAN FILE
# ============================================================

folder_project = Path(__file__).resolve().parent

nama_file = "image.jpg"
path_gambar = folder_project / nama_file

folder_output = folder_project / "output"
folder_output.mkdir(exist_ok=True)


# ============================================================
# 2. MEMBACA CITRA ASLI
# ============================================================

if not path_gambar.exists():
    raise FileNotFoundError(
        f"File {nama_file} tidak ditemukan!\n"
        "Pastikan image.jpg berada satu folder dengan main.py."
    )

img = imageio.imread(path_gambar)

# Mengubah citra RGB menjadi grayscale
if img.ndim == 3:

    if img.shape[2] == 4:
        img = img[:, :, :3]

    img = (
        0.299 * img[:, :, 0] +
        0.587 * img[:, :, 1] +
        0.114 * img[:, :, 2]
    )

img = np.clip(img, 0, 255).astype(np.float64)


# ============================================================
# 3. INFORMASI CITRA ASLI
# ============================================================

print("=" * 60)
print("PROJECT PCD SESI 5 - SPATIAL FILTERING")
print("=" * 60)

print("Nama citra          :", nama_file)
print("Ukuran citra        :", img.shape)
print("Nilai minimum       :", np.min(img))
print("Nilai maksimum      :", np.max(img))
print("Rata-rata intensitas:", np.mean(img))
print("Standar deviasi     :", np.std(img))


# ============================================================
# 4. MEAN FILTER 3x3 DAN 5x5
# ============================================================

mean_3x3 = ndimage.uniform_filter(
    img,
    size=3,
    mode="reflect"
)

mean_5x5 = ndimage.uniform_filter(
    img,
    size=5,
    mode="reflect"
)


# ============================================================
# 5. GAUSSIAN FILTER 3x3 DAN 5x5
# ============================================================

def buat_kernel_gaussian(ukuran, sigma):

    tengah = ukuran // 2

    sumbu = np.arange(-tengah, tengah + 1)

    x, y = np.meshgrid(sumbu, sumbu)

    kernel = np.exp(
        -(x**2 + y**2) / (2 * sigma**2)
    )

    kernel = kernel / np.sum(kernel)

    return kernel


# Gaussian Kernel 3x3
kernel_gaussian_3x3 = buat_kernel_gaussian(
    ukuran=3,
    sigma=0.8
)

# Gaussian Kernel 5x5
kernel_gaussian_5x5 = buat_kernel_gaussian(
    ukuran=5,
    sigma=1.2
)

gaussian_3x3 = ndimage.convolve(
    img,
    kernel_gaussian_3x3,
    mode="reflect"
)

gaussian_5x5 = ndimage.convolve(
    img,
    kernel_gaussian_5x5,
    mode="reflect"
)


# ============================================================
# 6. MEDIAN FILTER 3x3 DAN 5x5
# ============================================================

median_3x3 = ndimage.median_filter(
    img,
    size=3,
    mode="reflect"
)

median_5x5 = ndimage.median_filter(
    img,
    size=5,
    mode="reflect"
)


# ============================================================
# 7. SHARPENING FILTER
# ============================================================

kernel_sharpen = np.array([
    [ 0, -1,  0],
    [-1,  5, -1],
    [ 0, -1,  0]
], dtype=np.float64)

sharpening = ndimage.convolve(
    img,
    kernel_sharpen,
    mode="reflect"
)

sharpening = np.clip(
    sharpening,
    0,
    255
)


# ============================================================
# 8. MENGUMPULKAN HASIL FILTER
# ============================================================

hasil_filter = {

    "Citra Asli": img,

    "Mean Filter 3x3": mean_3x3,

    "Mean Filter 5x5": mean_5x5,

    "Gaussian Filter 3x3": gaussian_3x3,

    "Gaussian Filter 5x5": gaussian_5x5,

    "Median Filter 3x3": median_3x3,

    "Median Filter 5x5": median_5x5,

    "Sharpening": sharpening

}


# ============================================================
# 9. MENAMPILKAN NILAI INTENSITAS
# ============================================================

tengah_y = img.shape[0] // 2
tengah_x = img.shape[1] // 2

print("\n" + "=" * 60)
print("NILAI INTENSITAS SETIAP HASIL FILTER")
print("=" * 60)

data_intensitas = []

for nama, hasil in hasil_filter.items():

    minimum = np.min(hasil)
    maksimum = np.max(hasil)
    rata_rata = np.mean(hasil)
    standar_deviasi = np.std(hasil)

    piksel_tengah = hasil[tengah_y, tengah_x]

    perubahan = np.mean(np.abs(hasil - img))

    print("\n", nama)
    print("Minimum              :", minimum)
    print("Maksimum             :", maksimum)
    print("Rata-rata            :", rata_rata)
    print("Standar deviasi      :", standar_deviasi)
    print("Piksel tengah        :", piksel_tengah)
    print("Rata-rata perubahan  :", perubahan)

    data_intensitas.append([
        nama,
        minimum,
        maksimum,
        rata_rata,
        standar_deviasi,
        piksel_tengah,
        perubahan
    ])


# ============================================================
# 10. MENYIMPAN HASIL SETIAP FILTER
# ============================================================

for nama, hasil in hasil_filter.items():

    nama_gambar = (
        nama.lower()
        .replace(" ", "_")
        + ".png"
    )

    imageio.imwrite(
        folder_output / nama_gambar,
        np.clip(hasil, 0, 255).astype(np.uint8)
    )



# ============================================================
# 11. MEMBUAT GAMBAR PERBANDINGAN FILTER
# ============================================================

fig, axes = plt.subplots(
    3,
    3,
    figsize=(18, 15)
)

for ax, (nama, hasil) in zip(
    axes.flat,
    hasil_filter.items()
):

    ax.imshow(
        hasil,
        cmap="gray",
        vmin=0,
        vmax=255
    )

    ax.set_title(
        nama,
        fontsize=12,
        fontweight="bold",
        pad=12
    )

    ax.axis("off")

# Menghilangkan satu kotak kosong
axes[2, 2].axis("off")

# Judul utama
fig.suptitle(
    "PERBANDINGAN HASIL SPATIAL FILTERING",
    fontsize=18,
    fontweight="bold"
)

# Mengatur jarak antar gambar
plt.subplots_adjust(
    left=0.04,
    right=0.96,
    top=0.91,
    bottom=0.04,
    wspace=0.15,
    hspace=0.45
)

# Menyimpan gambar perbandingan
path_perbandingan = (
    folder_output /
    "hasil_spatial_filtering_lengkap.png"
)

plt.savefig(
    path_perbandingan,
    dpi=300,
    bbox_inches="tight"
)

plt.show()



# ============================================================
# 12. MENYIMPAN TABEL NILAI INTENSITAS KE CSV
# ============================================================

path_csv = (
    folder_output /
    "tabel_nilai_intensitas.csv"
)

with open(
    path_csv,
    "w",
    newline="",
    encoding="utf-8"
) as file:

    writer = csv.writer(file)

    writer.writerow([
        "Nama Filter",
        "Minimum",
        "Maksimum",
        "Rata-rata",
        "Standar Deviasi",
        "Intensitas Piksel Tengah",
        "Rata-rata Perubahan"
    ])

    writer.writerows(data_intensitas)


# ============================================================
# 13. MENYIMPAN RINGKASAN NILAI INTENSITAS TXT
# ============================================================

path_txt = (
    folder_output /
    "ringkasan_nilai_intensitas.txt"
)

with open(
    path_txt,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "PROJECT PCD SESI 5 - SPATIAL FILTERING\n"
    )

    file.write("=" * 60 + "\n")

    file.write("Nama citra: " + nama_file + "\n")

    file.write("Ukuran citra: " + str(img.shape) + "\n\n")

    for data in data_intensitas:

        file.write("\n" + data[0] + "\n")

        file.write(f"Minimum: {data[1]:.2f}\n")
        file.write(f"Maksimum: {data[2]:.2f}\n")
        file.write(f"Rata-rata: {data[3]:.2f}\n")
        file.write(f"Standar deviasi: {data[4]:.2f}\n")
        file.write(f"Piksel tengah: {data[5]:.2f}\n")
        file.write(f"Rata-rata perubahan: {data[6]:.2f}\n")


# ============================================================
# 14. INFORMASI AKHIR
# ============================================================

print("\n" + "=" * 60)
print("SELURUH PROSES FILTERING BERHASIL!")
print("=" * 60)

print("Citra input       :", nama_file)
print("Folder output     :", folder_output)
print("Gambar perbandingan:", path_perbandingan)
print("Tabel intensitas  :", path_csv)
print("Ringkasan intensitas:", path_txt)

print("\nSemua hasil berhasil disimpan!")
