# Analisis Performa CDCN dengan Supervisi Kedalaman dan Fine-tuning pada Dataset Pribadi

## Ringkasan penelitian

Repositori ini mendokumentasikan implementasi dan evaluasi **Central Difference Convolutional Network (CDCN)** untuk klasifikasi presentasi wajah asli dan serangan spoof. Penelitian menerapkan pelatihan awal pada dataset SiW, dilanjutkan dua skenario fine-tuning, yaitu menggunakan dataset pribadi saja dan menggunakan kombinasi dataset pribadi dengan data tambahan dari koleksi Real dan Display Spoof. Model diintegrasikan ke prototipe **Uji Coba CDCN Local**, yang menjalankan inferensi pada server laptop dan menampilkan hasil melalui peramban ponsel.

Metode mengacu pada CDCN dasar yang diperkenalkan oleh Yu dkk. pada CVPR 2020. Penelitian ini merupakan implementasi dan analisis empiris dengan penyesuaian pembagian data, prapemrosesan, dan konfigurasi komputasi; bukan pengusulan arsitektur baru maupun reproduksi persis seluruh protokol benchmark paper.

## 1. Komposisi dataset dan unit pengamatan

**Dataset pribadi utama yang digunakan dalam eksperimen bukan hanya dua video.** Manifest memuat **152 frame dari 12 kelompok video** yang dibagi menjadi data latih, validasi, dan uji. Angka `samples: 2` dalam JSON evaluasi menunjukkan dua video pada **subset uji**, bukan keseluruhan data pengembangan model. Jumlah kelompok video ditentukan dari pengelompokan rekaman pada manifest; jumlah tersebut tidak menyatakan jumlah orang yang berbeda.

### 1.1 Dataset pribadi utama

| Subset | Video asli | Video spoof | Total kelompok video | Frame asli tersedia | Frame spoof tersedia | Total frame tersedia |
|---|---:|---:|---:|---:|---:|---:|
| Latih | 4 | 4 | 8 | 35 | 56 | 91 |
| Validasi | 1 | 1 | 2 | 13 | 13 | 26 |
| Uji | 1 | 1 | 2 | 17 | 18 | 35 |
| **Jumlah** | **6** | **6** | **12** | **65** | **87** | **152** |

Frame yang tersedia merupakan kumpulan kandidat pada manifest. Jumlah frame ini berbeda dari jumlah yang digunakan pada satu epoch maupun satu proses evaluasi. Foto tambahan yang hubungan identitas atau sumber rekamannya belum terverifikasi tidak dimasukkan ke eksperimen ini. Rekaman uji tambahan dari eksperimen model biner terdahulu juga tidak digabungkan ke hasil model depth yang dilaporkan di bawah.

### 1.2 Skenario data latih gabungan

Data tambahan hanya digunakan pada subset latih. Subset validasi dan uji tetap sama dengan skenario dataset pribadi saja agar kedua hasil fine-tuning dapat dibandingkan pada data uji yang identik.

| Kondisi data gabungan | Kelompok video | Frame asli | Frame spoof | Total frame |
|---|---:|---:|---:|---:|
| Sebelum penyaringan deteksi wajah | 58 | 374 | 310 | 684 |
| Dikeluarkan karena wajah tidak terdeteksi | Tidak ada kelompok yang habis dikeluarkan | 0 | 5 | 5 |
| **Digunakan dalam manifest latih akhir** | **58** | **374** | **305** | **679** |

Jumlah 58 kelompok meliputi 34 kelompok asli dan 24 kelompok spoof. Berdasarkan manifest sumber, setiap kelompok spoof awal memiliki sedikitnya 7 frame. Karena seluruh 5 frame yang gagal merupakan spoof, pengecualian tersebut tidak dapat menghapus seluruh frame dari satu kelompok. Retensi jumlah kelompok ini merupakan inferensi dari manifest sumber dan ringkasan pengecualian; pemeriksaan ulang manifest depth akhir tetap dianjurkan untuk arsip reproduksi. Catatan hitungan tersedia pada [results/data_inventory.json](results/data_inventory.json).

### 1.3 Data SiW pada pelatihan awal

SiW digunakan melalui pembagian lokal yang telah disiapkan, bukan otomatis mengikuti seluruh protokol resmi benchmark.

| Subset SiW | Frame masukan | Frame lolos persiapan depth | Frame dikeluarkan |
|---|---:|---:|---:|
| Latih | 4.910 | 4.785 | 125 |
| Validasi | 1.247 | 1.162 | 85 |
| Uji | 1.429 | 1.390 | 39 |
| **Jumlah** | **7.586** | **7.337** | **249** |

Angka tersebut menggambarkan subset yang digunakan pada run SiW yang dilaporkan. Skrip yang lebih baru menambahkan fallback detektor, tetapi perubahan skrip tidak berarti hasil run SiW lama telah dihitung ulang. Gambar yang dikeluarkan dan cakupan deteksi harus tetap dilaporkan agar metrik pada subset yang lolos tidak disalahartikan sebagai keberhasilan seluruh alur pada semua masukan.

## 2. Pengolahan data dan supervisi model

1. **Pembentukan manifest.** Frame hasil ekstraksi rekaman dikelompokkan berdasarkan video asal dan diberi label `1` untuk asli serta `0` untuk spoof. Seluruh frame dari satu kelompok video dipertahankan dalam subset yang sama. Keunikan identitas orang dan hubungan antara rekaman asli dengan sumber replay masih memerlukan audit lebih lanjut.
2. **Deteksi wajah.** Pada persiapan depth terkini, sisi terpanjang gambar dibatasi 512 piksel. YuNet dijalankan pada confidence 0,80 dan, jika tidak menemukan wajah, dicoba kembali pada 0,65. Wajah dengan area terbesar dipilih. Kegagalan dicatat; tidak dibuat peta asli sintetis untuk menggantikan kegagalan rekonstruksi.
3. **Target kedalaman.** PRNet terlatih menghasilkan estimasi bentuk tiga dimensi wajah asli. Mesh dirasterisasi menjadi pseudo-depth yang selaras dengan RGB dan dinormalisasi ke rentang 0-1. Target spoof berupa peta nol. Pseudo-depth ini merupakan estimasi dari RGB, bukan pengukuran sensor kedalaman fisik.
4. **Transformasi berpasangan.** Crop dan pembalikan horizontal menerapkan koordinat geometris yang sama pada RGB dan target depth. Augmentasi warna, gamma, erasing, dan cutout diterapkan pada RGB dalam training. Validasi dan test tidak memakai augmentasi acak.
5. **Ukuran dan normalisasi.** RGB diubah menjadi 256 x 256 piksel dan target menjadi 32 x 32 piksel. Normalisasi masukan mengikuti `(piksel RGB - 127,5) / 128`. Faktor crop validasi/test adalah 1,3 terhadap lebar dan tinggi bbox.

Interval ekstraksi video awal tidak dapat disimpulkan hanya dari JSON hasil evaluasi. Untuk reproduksi penuh, skrip ekstraksi, manifest final, pemetaan video ke identitas, dan catatan pengecualian perlu diarsipkan bersama hasil. Dataset wajah tidak disertakan dalam repositori publik ini.

## 3. Algoritma dan rancangan pelatihan

CDCN dasar menggunakan Central Difference Convolution dengan `theta = 0,7`. Parameter theta mengatur operasi konvolusi dan berbeda dari threshold keputusan asli/spoof. Jaringan menerima RGB wajah dan memprediksi peta kedalaman. Fungsi objektif merupakan penjumlahan **mean squared error (MSE)** dan **contrastive depth loss (CDL)**.

| Parameter | Pelatihan awal SiW | Fine-tuning dataset pribadi saja | Fine-tuning gabungan |
|---|---|---|---|
| Inisialisasi | Bobot acak | `best.pth` SiW depth | `best.pth` SiW depth yang sama |
| Optimizer | Adam | Adam | Adam |
| Learning rate awal | 0,0001 | 0,00001 | 0,00001 |
| Weight decay | 0,00005 | 0,00005 | 0,00005 |
| Batch fisik / akumulasi | 8 / 7 | 8 / 7 | 8 / 7 |
| Epoch yang tercatat selesai | 149 | 50 | 50 |
| Epoch checkpoint terbaik yang digunakan | 119 | 1 | 1 |

Konfigurasi awal SiW menyediakan maksimum 1.300 epoch. Pada pelaksanaan pengguna, batas kemudian diubah menjadi 200 dan checkpoint terakhir yang tercatat berada pada epoch 149. Oleh karena itu, 1.300 maupun 200 tidak boleh ditulis sebagai jumlah epoch yang telah diselesaikan. Akumulasi 7 batch menghasilkan paling banyak 56 sampel per pembaruan untuk jendela penuh; jendela terakhir dapat lebih kecil. Statistik BatchNorm tetap mengikuti batch fisik, bukan batch akumulasi.

### 3.1 Pengambilan frame saat training

Setiap epoch memilih **satu frame acak dari setiap kelompok video training**. Dengan demikian, manifest 91 frame pada skenario pribadi saja menyediakan kandidat dari 8 video, tetapi satu epoch menggunakan 8 frame terpilih. Pada skenario gabungan, satu epoch mengambil satu frame dari masing-masing 58 kelompok video. Frame yang terpilih dapat berubah antar-epoch dan tidak dijamin seluruh kandidat terpakai selama 50 epoch. Strategi ini memberi kontribusi yang seimbang per video, bukan per frame. Strategi tersebut merupakan pilihan implementasi eksperimen, bukan ketentuan wajib arsitektur CDCN.

### 3.2 Validasi, threshold, dan pemilihan checkpoint

Validasi mengambil maksimal 8 frame unik yang tersebar menurut indeks frame pada setiap video. Threshold ditentukan pada titik EER empiris diskret dari skor video validasi, lalu model dipilih berdasarkan ACER validasi terendah. Titik tersebut mendekati keseimbangan APCER dan BPCER; keduanya tidak harus persis sama pada sampel diskret.

Pada kedua fine-tuning, validasi hanya mencakup 1 video asli dan 1 video spoof, dengan total 16 frame terpilih. Riwayat menunjukkan ACER validasi telah mencapai 0% pada epoch pertama. Kode mengganti `best.pth` hanya jika ACER benar-benar menurun (`<`), sehingga checkpoint terbaik tetap berasal dari **epoch 1**, meskipun training berlanjut hingga 50 epoch. `latest.pth` menyimpan epoch 50. Dengan demikian, hasil test berikut merupakan hasil checkpoint yang dipilih berdasarkan aturan validasi tersebut, bukan otomatis bobot epoch terakhir.

## 4. Protokol evaluasi dan definisi metrik

Untuk satu frame, skor diperoleh dari rata-rata seluruh nilai peta kedalaman prediksi. Untuk satu video, skor merupakan rata-rata skor maksimal 8 frame terpilih. Video diterima sebagai asli jika skor video mencapai atau melebihi threshold validasi. Threshold tidak dioptimalkan menggunakan label test.

- **Accuracy** = jumlah video dengan prediksi benar / seluruh video uji x 100%.
- **APCER** = jumlah video spoof diterima sebagai asli / seluruh video spoof uji x 100%.
- **BPCER** = jumlah video asli ditolak sebagai spoof / seluruh video asli uji x 100%.
- **ACER** = (APCER + BPCER) / 2.

APCER pada implementasi ini dihitung secara agregat atas seluruh video spoof dalam subset. Pelaporan per jenis media serangan belum tersedia. Nilai loss merupakan kesalahan prediksi peta kedalaman pada frame yang dievaluasi, bukan persentase kesalahan klasifikasi. Skor CDCN juga bukan probabilitas keyakinan yang telah dikalibrasi.

### 4.1 Ketersediaan frame dan sampel yang benar-benar dievaluasi

| Subset uji | Frame tersedia setelah penyaringan | Frame yang diproses evaluator | Video sebagai unit metrik | Video asli / spoof |
|---|---:|---:|---:|---:|
| SiW | 1.390 | 929 | 130 | 72 / 58 |
| Dataset pribadi | 35 | 16 | 2 | 1 / 1 |

Seluruh 35 frame pribadi tetap tercatat pada manifest test, tetapi konfigurasi evaluator memilih 8 frame dari masing-masing dua video. Oleh karena itu, 16 frame bukan 16 percobaan video independen; keluaran `samples: 2` memang konsisten dengan evaluasi berbasis video.

## 5. Hasil empiris

### 5.1 Model SiW depth sebelum adaptasi

| Unit uji | Threshold validasi | Spoof diterima | Asli ditolak | Accuracy | APCER | BPCER | ACER | Loss |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 130 video SiW | 0,535589 | 0/58 | 11/72 | 91,54% | 0,00% | 15,28% | 7,64% | 0,09029 |

Hasil ini menunjukkan model menolak seluruh video spoof pada subset SiW yang diuji, tetapi menolak 11 dari 72 video asli. Hasil SiW berasal dari keluaran Colab yang ditranskripsi dan diberi penanda pada [results/test_siw_transcribed.json](results/test_siw_transcribed.json).

### 5.2 Perbandingan dua skenario fine-tuning pada subset uji pribadi yang sama

| Skenario fine-tuning | Frame latih tersedia / kelompok video | Threshold validasi | Spoof diterima pada test | Asli ditolak pada test | Accuracy | APCER | BPCER | ACER | Loss test |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Dataset pribadi saja | 91 / 8 | 0,553362 | 0/1 | 0/1 | 100,00% | 0,00% | 0,00% | 0,00% | 0,25414 |
| Dataset pribadi + data tambahan | 679 / 58 | 0,522483 | 0/1 | 1/1 | 50,00% | 0,00% | 100,00% | 50,00% | 0,22746 |

Kedua baris dievaluasi pada **2 video test yang sama**, dengan 16 frame terpilih. Nilai threshold ditampilkan hingga enam desimal; evaluator menggunakan nilai lengkap yang tersimpan pada checkpoint. Berkas asli hasil test setelah penyeragaman nama tersedia pada [results/test_pribadi_only.json](results/test_pribadi_only.json) dan [results/test_pribadi_mixed.json](results/test_pribadi_mixed.json).

### 5.3 Interpretasi

Pada subset uji pribadi yang tersedia, checkpoint fine-tuning dataset pribadi saja menghasilkan klasifikasi benar untuk satu video asli dan satu video spoof. Checkpoint gabungan menolak video spoof dengan benar, tetapi salah menolak video asli. Dengan demikian, skenario pribadi saja memberikan ACER lebih rendah **pada pengujian terbatas ini** dan dipilih sementara untuk prototipe web.

Loss test gabungan lebih rendah daripada skenario pribadi saja, tetapi keputusan klasifikasinya lebih buruk pada threshold validasinya. Hal ini menunjukkan bahwa penurunan loss regresi kedalaman tidak selalu sejalan dengan penurunan kesalahan klasifikasi pada operating point yang digunakan.

Belum dapat disimpulkan bahwa penambahan data menyebabkan penurunan performa secara umum atau bahwa salah satu model unggul secara signifikan. Kemungkinan pengaruh perbedaan domain, ketidakseimbangan kontribusi sumber, karakteristik sampel uji, dan kestabilan seleksi checkpoint perlu diuji melalui eksperimen lanjutan. Hasil SiW dan hasil pribadi menggunakan test berbeda, sehingga perbedaan persentase antar-dataset bukan bukti peningkatan langsung akibat fine-tuning.

## 6. Keterbatasan dan rekomendasi pengembangan

1. **Ukuran validasi dan test terbatas.** Setiap subset pribadi hanya memiliki satu video per kelas. Satu kesalahan mengubah APCER atau BPCER sebesar 100 poin persentase. Angka accuracy 100% dan ACER 0% harus dibaca sebagai hasil dua rekaman tersebut, bukan estimasi keandalan umum.
2. **Independensi data perlu diverifikasi.** Pemeriksaan kode mendeteksi overlap path, video ID, subject ID yang terisi, dan gambar identik lintas subset. Pemeriksaan tersebut belum membuktikan tidak ada orang yang sama, rekaman turunan, atau sumber replay yang sama ketika metadata identitas tidak tersedia. Buat tabel identitas dan hubungan sumber asli-spoof sebelum menyatakan protokol subject-disjoint.
3. **Seleksi checkpoint masih kasar.** Validasi dua video menghasilkan banyak epoch dengan ACER sama. Untuk eksperimen berikutnya, perluas validasi dan tetapkan aturan tie-break sebelum training, misalnya memakai loss validasi sebagai kriteria sekunder. Perubahan aturan merupakan eksperimen baru dan tidak mengubah arti hasil yang telah dilaporkan.
4. **Belum ada pembanding SiW awal pada test pribadi yang sama.** Evaluasi ulang checkpoint SiW depth pada subset pribadi dengan threshold dari validasi yang ditentukan sesuai protokol yang dinyatakan. Nyatakan apakah tujuannya transfer tanpa kalibrasi atau adaptasi threshold. Hasilnya diperlukan untuk mengukur dampak fine-tuning pada domain pribadi secara langsung.
5. **Test telah memengaruhi pemilihan prototipe.** Karena checkpoint prototipe dipilih setelah melihat hasil test ini, gunakan rekaman baru yang belum digunakan untuk memilih model atau threshold sebagai evaluasi akhir independen.
6. **Perlu variasi dan pengulangan.** Tambahkan rekaman independen dari beberapa orang, sesi, pencahayaan, jarak, dan jenis media spoof. Jumlah frame dari satu video tidak menggantikan jumlah rekaman independen. Setelah tersedia cukup kelompok, laporkan pengulangan dengan seed atau split terkontrol dan variasi hasilnya.
7. **Evaluasi aplikasi berbeda dari evaluasi model.** Web menilai per frame, sedangkan metrik di atas per video. Tetapkan protokol uji aplikasi yang mencatat label sebenarnya, false acceptance/rejection, kegagalan deteksi, dan waktu respons. Jangan menyamakan kotak hijau/merah dengan kebenaran label.

**Kesimpulan terbatas yang didukung data:** pada dua video uji pribadi yang digunakan, checkpoint CDCN fine-tuning dataset pribadi saja menghasilkan accuracy 100% dan ACER 0%, sedangkan checkpoint gabungan menghasilkan accuracy 50% dan ACER 50%. Temuan ini mendukung pemilihan awal prototipe, tetapi belum cukup untuk menyimpulkan keunggulan umum atau keandalan pada populasi dan kondisi pengujian yang lebih luas.

## Isi penting
- models/CDCNs.py dan models/loss.py: core CDCN dan loss.
- prepare_depth.py: target PRNet dan bbox YuNet.
- datasets/DepthDataset.py: data, sampling, transform berpasangan.
- trainer/depth_training.py dan utils/depth_metrics.py: training, threshold dan metrik.
- uji_cdcn_local.py dan uji_cdcn_local/index.html: kamera live/upload video.
- config/, tests/, results/: konfigurasi, pemeriksaan, hasil.
- docs/NASKAH_VIDEO.md: narasi lengkap dan petunjuk rekaman.
- docs/PANDUAN_GITHUB.md: file unggahan dan langkah GitHub.

## Instalasi lokal
Python3.11, Windows PowerShell:
```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt -r requirements-lab.txt
.\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_*depth*.py" -v
```
Gunakan Python3.11 terpisah dengan requirements-depth.txt untuk PRNet agar dependensi TensorFlow teacher terpisah dari PyTorch.

## Web
Sediakan aset sesuai checkpoints/README.md. Tidak ada bobot/data dalam Git.
```powershell
.\.venv\Scripts\python.exe uji_cdcn_local.py --host 0.0.0.0 --port 5000
```
Untuk kamera HP gunakan HTTPS. Ganti IPv4 contoh dengan IP laptop saat ini:
```powershell
.\.venv\Scripts\python.exe uji_cdcn_local_tls.py 192.168.1.29
.\.venv\Scripts\python.exe uji_cdcn_local.py --host 0.0.0.0 --port 5443 --tls-cert checkpoints/tls/server.pem --tls-key checkpoints/tls/server-key.pem
```
HP dan laptop pada Wi-Fi sama. Pasang CA publik lokal checkpoints/tls/Uji-CDCN-Local-CA.cer pada HP jika diperlukan; jangan membagikan server-key.pem. Membuat sertifikat baru mengganti CA lama. Izinkan akses jaringan lokal proses server bila firewall meminta. Buka https://IP-LAPTOP:5443 dan izinkan kamera.

Profil web: checkpoint depth dataset pribadi saja, threshold awal0.553362, sisi gambar maksimum512, deteksi0.8/fallback0.65, wajah terbesar, crop persegi panjang1.3x, resize256x256, RGB, normalisasi(pixel-127.5)/128. Slider untuk eksplorasi; test formal memakai threshold validasi tersimpan. PRNet tidak berjalan saat inferensi kamera.

## Training/evaluasi
Notebook disertakan tanpa output. Notebook mengharapkan paket privat CDCN-PRNet-Depth-Colab.zip di Drive; paket data tidak ada dalam Git. Notebook pemulihan juga memakai pribadi_prepared.zip dan best.pth SiW. Untuk data sendiri, siapkan CSV path,label (1 asli,0 spoof), subject_id jika diketahui, dan pengelompokan video/sumber yang benar. Lihat docs/METODE_DAN_HASIL.md.
```bash
python prepare_depth.py --root data/siw --namespace siw --csv train.csv val.csv test.csv
python train.py --config config/CDCN_depth.yaml --check-data
python train.py --config config/CDCN_depth.yaml
python train.py --config config/CDCN_depth_pribadi_only.yaml --init-checkpoint /path/siw/best.pth
python evaluate.py --config config/CDCN_depth_pribadi_only.yaml --checkpoint /path/pribadi_only/best.pth --test-csv test_depth.csv
```
Sesuaikan dataset root dan output_dir/log_dir YAML. Default SiW max1300 epoch; run dilaporkan mencapai149, best119. Konfigurasi maksimum bukan jumlah epoch aktual. Source legacy masih diimpor entrypoint untuk kompatibilitas; pipeline default prnet_depth, legacy harus diaktifkan eksplisit.

## Referensi dan video
Lihat THIRD_PARTY_NOTICES.md. Tambahkan tautan YouTube setelah unggah. Nama peneliti dan tautan repository diisi pemilik.
