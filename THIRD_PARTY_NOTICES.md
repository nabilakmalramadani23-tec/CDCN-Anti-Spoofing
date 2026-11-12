# Sumber dan atribusi
Proyek ini merupakan adaptasi eksperimen, bukan pengusulan arsitektur CDCN baru.
- Yu dkk., Searching Central Difference Convolutional Networks for Face Anti-Spoofing, CVPR2020. https://arxiv.org/abs/2003.04092
- Kode CDCN: https://github.com/ZitongYu/CDCN/tree/master/CVPR2020_paper_codes ; snapshot referensi di third_party/reference/CDCNs.py. Commit dan hash di docs/depth_sources.json.
- Predictor PRNet: https://github.com/YadiraF/PRNet ; lisensi MIT sumber dipertahankan di third_party/prnet/LICENSE.
- YuNet: https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet ; bobot tidak disertakan.
Implementasi target depth, loader berpasangan, training/evaluasi lokal, dan web merupakan adaptasi eksperimen. Lisensi PRNet hanya berlaku pada komponen terkait; paket ini tidak menyatakan seluruh kode pihak ketiga berlisensi MIT. Ikuti ketentuan sumber masing-masing untuk penggunaan/distribusi ulang.
