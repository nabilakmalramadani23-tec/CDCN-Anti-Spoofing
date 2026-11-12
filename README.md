# Analisis Performa CDCN dan Fine-tuning Dataset Pribadi
Simulasi core algoritma face anti-spoofing dengan **CDCN dasar (theta 0.7)**, supervisi peta kedalaman PRNet, dan web Uji Coba CDCN Local untuk pengujian kamera HP melalui server laptop. Implementasi ini mengadaptasi metode Yu dkk.

## Alur metode
1. Siapkan train/val/test. Pertahankan satu video dan rekaman terkait pada satu split; audit identitas/sumber replay Dataset Pribadi masih diperlukan.
2. PRNet menghasilkan pseudo-depth wajah asli; spoof diberi peta nol.
3. RGB/depth memakai crop/flip geometris selaras. Input RGB256x256 dan target32x32.
4. Train dengan MSE + contrastive depth loss, satu frame acak per video setiap epoch.
5. Validasi/test memilih maksimal8 frame/video dan merata-ratakan skor per video. Threshold dipilih dari EER empiris validasi.
6. HP mengirim frame ke laptop; web menampilkan keputusan per frame, skor, dan bounding box.

## Hasil
| Model | Test | Accuracy | APCER | BPCER | ACER |
|---|---|---:|---:|---:|---:|
| SiW depth awal |130 video SiW|91.54%|0%|15.28%|7.64%|
| dataset pribadi saja |2 video Dataset Pribadi|100%|0%|0%|0%|
| Gabungan |2 video Dataset Pribadi yang sama|50%|0%|100%|50%|
SiW dan Dataset Pribadi memakai test berbeda, bukan bukti peningkatan langsung. Test Dataset Pribadi hanya1 asli+1 spoof dan16 frame terpilih. Angka100% bukan jaminan keandalan umum. Skor bukan probabilitas terkalibrasi.

Kedua fine-tuning selesai50 epoch. Best yang diuji berasal dari epoch1: best diganti hanya bila ACER validasi turun; pada kedua run ACER sudah0% di awal. Latest berasal dari epoch50. Penurunan loss tidak otomatis mengganti best.
Hasil Dataset Pribadi berasal dari JSON test dan riwayat ZIP checkpoint pengguna. Hasil SiW ditranskripsi dari keluaran Colab dan ditandai sebagai transkripsi. Lihat results/.

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
