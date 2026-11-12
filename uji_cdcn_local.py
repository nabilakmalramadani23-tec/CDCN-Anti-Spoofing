import argparse
from io import BytesIO
import math
from pathlib import Path
import tempfile
from threading import Lock

import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request
from PIL import Image
import torch
from waitress import serve
from werkzeug.serving import run_simple

from datasets.DepthDataset import NORMALIZATION, PairedDepthTransform
from utils.eval import predict
from utils.utils import build_network, get_device, read_cfg


ROOT = Path(__file__).resolve().parent
MAX_SECONDS = 60
MAX_UPLOAD = 64 * 1024 * 1024


def score_frame(frame, detector, model, transform, device):
    height, width = frame.shape[:2]
    scale = min(512 / max(height, width), 1)
    small = cv2.resize(frame, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_AREA)
    detector.setInputSize((small.shape[1], small.shape[0]))
    detector.setScoreThreshold(0.80)
    _, faces = detector.detect(small)
    if faces is None:
        detector.setScoreThreshold(0.65)
        _, faces = detector.detect(small)
    if faces is None:
        return dict(score=None, status='no_face')
    x, y, bw, bh = (float(v) for v in max(faces, key=lambda f: f[2] * f[3])[:4])
    # Reuse validation's exact rectangular 1.3 crop, resize, RGB order and normalization.
    rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
    tensor, _ = transform(rgb, np.zeros(small.shape[:2], dtype=np.float32), (x, y, bw, bh))
    with torch.inference_mode():
        depth_map = model(tensor.unsqueeze(0).to(device))[0]
        _, score = predict(depth_map)
    height, width = small.shape[:2]
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(width, x + bw), min(height, y + bh)
    return dict(score=score.item(), status='scored',
                box=[x1 / width, y1 / height, (x2 - x1) / width, (y2 - y1) / height])


def score_video(path, detector, model, transform, device):
    cap = cv2.VideoCapture(str(path))
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if not cap.isOpened() or not math.isfinite(fps) or fps <= 0 or count < 1:
            raise ValueError('Video tidak bisa dibaca. Coba rekaman MP4.')
        if count / fps > MAX_SECONDS:
            raise ValueError(f'Pilih video dengan durasi maksimal {MAX_SECONDS} detik.')

        samples = []
        for second in range(math.ceil(count / fps)):
            timestamp = second + 0.5
            index = round(timestamp * fps)
            if index >= count:
                break
            cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = cap.read()
            if not ok:
                samples.append(dict(time=timestamp, score=None, status='decode'))
                continue
            samples.append(dict(time=timestamp, **score_frame(frame, detector, model, transform, device)))
        return samples
    finally:
        cap.release()


def create_app(checkpoint, detector_path, cfg):
    threshold = float(cfg['val']['threshold'])
    if not 0 <= threshold <= 1:
        raise ValueError('Threshold awal harus berada di rentang 0–1.')
    device = get_device(cfg)
    model = build_network(cfg).to(device)
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    if state.get('supervision') != 'prnet_depth' or state.get('normalization') != NORMALIZATION:
        raise ValueError('Uji Coba CDCN Local requires a PRNet-depth CDCN checkpoint with matching normalization.')
    model.load_state_dict(state['state_dict'])
    model.eval()
    transform = PairedDepthTransform(training=False)
    detector = cv2.FaceDetectorYN.create(str(detector_path), '', (640, 640), 0.80, 0.3, 5000)
    inference_lock = Lock()
    app = Flask(__name__, template_folder='uji_cdcn_local')
    app.config['MAX_CONTENT_LENGTH'] = MAX_UPLOAD

    @app.get('/')
    def index():
        return render_template('index.html', checkpoint=checkpoint.name, device=str(device),
                               threshold=f'{threshold:.6f}')

    @app.get('/health')
    def health():
        return jsonify(checkpoint=checkpoint.name, device=str(device), threshold=threshold,
                       supervision=state['supervision'], normalization=NORMALIZATION,
                       checkpoint_epoch=state['epoch'] + 1,
                       validation_threshold=state['threshold'], crop_scale=1.3,
                       detector_max_side=512, score_unit='frame')

    @app.errorhandler(413)
    def too_large(_error):
        return jsonify(error='Ukuran unggahan melebihi batas 64 MB.'), 413

    @app.post('/analyze')
    def analyze():
        video = request.files.get('video')
        if video is None or not video.filename:
            return jsonify(error='Pilih atau rekam video terlebih dahulu.'), 400
        suffix = Path(video.filename).suffix.lower()
        if suffix not in {'.mp4', '.mov', '.3gp', '.webm', '.m4v', '.avi', '.mkv'}:
            return jsonify(error='Format video belum didukung. Coba MP4.'), 400
        with tempfile.TemporaryDirectory(prefix='cdcn-video-') as tmp:
            path = Path(tmp) / f'video{suffix}'
            video.save(path)
            try:
                with inference_lock:
                    samples = score_video(path, detector, model, transform, device)
            except ValueError as exc:
                return jsonify(error=str(exc)), 422
        return jsonify(samples=samples)

    @app.post('/analyze-frame')
    def analyze_frame():
        if request.mimetype != 'image/jpeg':
            return jsonify(error='Frame kamera harus berupa JPEG.'), 415
        request.max_content_length = 512 * 1024
        raw = request.get_data()
        if not raw:
            return jsonify(error='Frame kamera kosong.'), 400
        try:
            with Image.open(BytesIO(raw)) as image:
                if image.format != 'JPEG' or max(image.size) > 640:
                    return jsonify(error='Frame kamera melebihi ukuran yang didukung.'), 422
            frame = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
        except (OSError, cv2.error, Image.DecompressionBombError):
            frame = None
        if frame is None:
            return jsonify(error='Frame kamera tidak bisa dibaca.'), 422
        with inference_lock:
            result = score_frame(frame, detector, model, transform, device)
        return jsonify(result)

    return app


def main():
    parser = argparse.ArgumentParser(description='Uji Coba CDCN Local')
    parser.add_argument('--config', type=Path, default=ROOT / 'config/CDCN_pribadi_local.yaml')
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--detector', type=Path)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', default=5000, type=int)
    parser.add_argument('--tls-cert', type=Path)
    parser.add_argument('--tls-key', type=Path)
    args = parser.parse_args()
    if bool(args.tls_cert) != bool(args.tls_key):
        parser.error('--tls-cert dan --tls-key harus dipakai bersama')
    cv2.setNumThreads(2)
    torch.set_num_threads(2)
    cfg = read_cfg(args.config)
    checkpoint = args.checkpoint or ROOT / cfg['checkpoint']
    detector = args.detector or ROOT / cfg['face_detector']
    app = create_app(checkpoint, detector, cfg)
    if args.tls_cert:
        run_simple(args.host, args.port, app, threaded=True,
                   ssl_context=(str(args.tls_cert), str(args.tls_key)))
        return
    print(f'Uji Coba CDCN Local: http://{args.host}:{args.port}', flush=True)
    addresses = dict.fromkeys(('127.0.0.1', args.host))
    serve(app, listen=' '.join(f'{addr}:{args.port}' for addr in addresses),
          max_request_body_size=MAX_UPLOAD)


if __name__ == '__main__':
    main()
