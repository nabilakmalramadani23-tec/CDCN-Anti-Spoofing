"""Generate image-aligned depth supervision using the official pretrained PRNet.

The original TensorFlow predictor is retained in third_party/prnet. Compatibility
changes preserve its checkpoint variables and inference BatchNorm equations.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import re

import cv2
import numpy as np
from numba import njit


ROOT = Path(__file__).resolve().parent


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def load_prnet(folder):
    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
    import tensorflow.compat.v1 as tf
    import tf_slim

    tf.disable_v2_behavior()

    @tf_slim.add_arg_scope
    def batch_norm(inputs, is_training=False, scale=True, scope=None, **_kwargs):
        if is_training:
            raise ValueError('PRNet is used as a fixed pretrained depth generator')
        channels = inputs.shape.as_list()[-1]
        with tf.variable_scope(scope or 'BatchNorm'):
            beta = tf.get_variable('beta', [channels], initializer=tf.zeros_initializer())
            gamma = tf.get_variable('gamma', [channels], initializer=tf.ones_initializer()) if scale else None
            mean = tf.get_variable('moving_mean', [channels], initializer=tf.zeros_initializer(), trainable=False)
            variance = tf.get_variable('moving_variance', [channels], initializer=tf.ones_initializer(), trainable=False)
            return tf.nn.batch_normalization(inputs, mean, variance, beta, gamma, 0.001)

    source = (ROOT / 'third_party/prnet/predictor.py').read_text(encoding='utf-8')
    source = source.replace('import tensorflow as tf', 'import tensorflow.compat.v1 as tf')
    source = source.replace('import tensorflow.contrib.layers as tcl', 'import tf_slim as tcl')
    source = source.replace('from tensorflow.contrib.framework import arg_scope', 'from tf_slim import arg_scope')
    source = source.replace('tcl.batch_norm', '_reference_batch_norm')
    source = source.replace('num_outputs/2', 'num_outputs//2')
    namespace = {'_reference_batch_norm': batch_norm, '__name__': 'prnet_compat'}
    exec(compile(source, 'third_party/prnet/predictor.py', 'exec'), namespace)
    predictor = namespace['PosPrediction']()
    predictor.restore(str(folder / '256_256_resfcn256_weight'))
    return predictor


@njit(cache=True)
def rasterize(vertices, triangles, height, width):
    """Image-coordinate z buffer, barycentric triangle interpolation, nearer z wins."""
    depth = np.zeros((height, width), dtype=np.float32)
    buffer = np.full((height, width), -np.inf, dtype=np.float32)
    for triangle in triangles:
        a, b, c = vertices[triangle[0]], vertices[triangle[1]], vertices[triangle[2]]
        denominator = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1])
        if abs(denominator) < 1e-8:
            continue
        left = max(0, int(np.ceil(min(a[0], b[0], c[0]))))
        right = min(width - 1, int(np.floor(max(a[0], b[0], c[0]))))
        top = max(0, int(np.ceil(min(a[1], b[1], c[1]))))
        bottom = min(height - 1, int(np.floor(max(a[1], b[1], c[1]))))
        for y in range(top, bottom + 1):
            for x in range(left, right + 1):
                w0 = ((b[1] - c[1]) * (x - c[0]) + (c[0] - b[0]) * (y - c[1])) / denominator
                w1 = ((c[1] - a[1]) * (x - c[0]) + (a[0] - c[0]) * (y - c[1])) / denominator
                w2 = 1 - w0 - w1
                if w0 >= -1e-6 and w1 >= -1e-6 and w2 >= -1e-6:
                    z = w0 * a[2] + w1 * b[2] + w2 * c[2]
                    if z > buffer[y, x]:
                        buffer[y, x] = z
                        depth[y, x] = z
    return depth, np.isfinite(buffer)


def face_depth(image, bbox, predictor, face_indices, triangles):
    x, y, width, height = bbox
    size = int((width + height) / 2 * 1.6)
    if size <= 1:
        raise ValueError('Invalid PRNet crop')
    center = np.array([x + width / 2, y + height / 2], dtype=np.float32)
    points = np.array([center - size / 2, center + [-size / 2, size / 2],
                       center + [size / 2, -size / 2]], dtype=np.float32)
    target = np.array([[0, 0], [0, 255], [255, 0]], dtype=np.float32)
    matrix = cv2.getAffineTransform(points, target)
    cropped = cv2.warpAffine(image.astype(np.float32) / 255.0, matrix, (256, 256))
    positions = predictor.predict(cropped).reshape(-1, 3)
    if not np.isfinite(positions).all():
        raise ValueError('PRNet predicted nonfinite vertices')
    positions[:, 2] /= matrix[0, 0]
    positions[:, :2] = cv2.transform(positions[None, :, :2], cv2.invertAffineTransform(matrix))[0]
    vertices = positions[face_indices].astype(np.float32)
    low, high = vertices[:, 2].min(), vertices[:, 2].max()
    if high - low <= 1e-5:
        raise ValueError('PRNet predicted flat geometry')
    vertices[:, 2] = (vertices[:, 2] - low) / (high - low)
    depth, mask = rasterize(vertices, triangles, image.shape[0], image.shape[1])
    if mask.sum() < 100 or depth[mask].std() <= 1e-5:
        raise ValueError('PRNet face mesh has insufficient image coverage')
    return np.rint(np.clip(depth, 0, 1) * 255).astype(np.uint8)


def video_id(row, path, namespace):
    if row.get('video_id'):
        return namespace + '/' + row['video_id']
    if path.parent.name in {'real', 'spoof', 'images'}:
        return namespace + '/' + path.stem.rsplit('_', 1)[0]
    return namespace + '/' + path.parent.name


def prepare(args):
    root = args.root.resolve()
    folder = args.prnet_dir.resolve()
    data_file = folder / '256_256_resfcn256_weight.data-00000-of-00001'
    for file in (data_file, folder / '256_256_resfcn256_weight.index',
                 folder / 'face_ind.txt', folder / 'triangles.txt', args.detector):
        if not file.is_file():
            raise FileNotFoundError(f'Missing pretrained PRNet/YuNet asset: {file}')
    provenance = dict(generator='official_PRNet_pretrained', weights_sha256=digest(data_file),
                      predictor_sha256=digest(ROOT / 'third_party/prnet/predictor.py'),
                      face_indices_sha256=digest(folder / 'face_ind.txt'),
                      triangles_sha256=digest(folder / 'triangles.txt'),
                      detector_sha256=digest(args.detector), max_image_side=512,
                      detection_policy={'primary': 0.8, 'fallback': 0.65})
    saved = root / 'depth_provenance.json'
    if saved.exists():
        previous = json.loads(saved.read_text())
        legacy = {key: value for key, value in provenance.items() if key != 'detection_policy'}
        if previous not in (provenance, legacy):
            raise ValueError('Depth cache provenance differs. Use a separate dataset directory for this generator.')
    detector = cv2.FaceDetectorYN.create(str(args.detector), '', (512, 512), 0.8, 0.3, 5000)
    face_indices = np.loadtxt(folder / 'face_ind.txt').astype(np.int32)
    triangles = np.loadtxt(folder / 'triangles.txt').astype(np.int32)
    predictor = load_prnet(folder)
    failures, summary = [], {}
    if not saved.exists() and (root / 'prepared').exists():
        raise ValueError('Prepared files have no provenance. Start with a fresh dataset directory.')
    saved.write_text(json.dumps(provenance, indent=2))
    columns = ['path', 'label', 'depth_path', 'video_id', 'subject_id', 'frame_index',
               'bbox_x', 'bbox_y', 'bbox_w', 'bbox_h', 'detector_score', 'detection_threshold']
    cache = {}
    for name in args.csv:
        with (root / name).open(encoding='utf-8-sig', newline='') as stream:
            rows = list(csv.DictReader(stream))
        if args.limit_per_class:
            rows = [row for label in ('0', '1') for row in
                    [item for item in rows if item['label'] == label][:args.limit_per_class]]
        output = []
        for index, row in enumerate(rows):
            path = Path(row['path'])
            try:
                key = (str(path), row['label'])
                if key not in cache:
                    image = cv2.imread(str(root / path))
                    if image is None:
                        raise ValueError('Cannot decode source image')
                    height, width = image.shape[:2]
                    scale = min(512 / max(height, width), 1)
                    image = cv2.resize(image, (round(width * scale), round(height * scale)),
                                       interpolation=cv2.INTER_AREA)
                    detector.setInputSize((image.shape[1], image.shape[0]))
                    detection_threshold = 0.8
                    detector.setScoreThreshold(detection_threshold)
                    _, faces = detector.detect(image)
                    if faces is None:
                        detection_threshold = 0.65
                        detector.setScoreThreshold(detection_threshold)
                        _, faces = detector.detect(image)
                    if faces is None:
                        raise ValueError('Face detector found no face')
                    face = max(faces, key=lambda f: f[2] * f[3])
                    bbox = [float(v) for v in face[:4]]
                    identity = hashlib.sha256((str(path) + digest(root / path)).encode()).hexdigest()
                    rgb_path = Path('prepared/rgb') / (identity + '.png')
                    depth_path = Path('prepared/depth') / (identity + '.png')
                    (root / rgb_path).parent.mkdir(parents=True, exist_ok=True)
                    (root / depth_path).parent.mkdir(parents=True, exist_ok=True)
                    if row['label'] == '1' and not (root / depth_path).is_file():
                        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        depth = face_depth(rgb, bbox, predictor, face_indices, triangles)
                        ok, encoded = cv2.imencode('.png', depth)
                        if not ok:
                            raise ValueError('Cannot encode depth map')
                        temporary = (root / depth_path).with_suffix('.tmp')
                        temporary.write_bytes(encoded.tobytes())
                        temporary.replace(root / depth_path)
                    ok, encoded = cv2.imencode('.png', image)
                    if not ok:
                        raise ValueError('Cannot encode RGB')
                    (root / rgb_path).write_bytes(encoded.tobytes())
                    match = re.search(r'(\d+)(?:ms)?$', path.stem)
                    if match is None:
                        raise ValueError('Source filename needs a frame number or millisecond timestamp')
                    cache[key] = dict(path=rgb_path.as_posix(), label=row['label'],
                                      depth_path=depth_path.as_posix() if row['label'] == '1' else '',
                                      video_id=video_id(row, path, args.namespace),
                                      subject_id=row.get('subject_id', ''),
                                      frame_index=int(match[1]),
                                      detector_score=float(face[-1]), detection_threshold=detection_threshold,
                                      **{f'bbox_{key}': value for key, value in zip(('x', 'y', 'w', 'h'), bbox)})
                output.append(cache[key])
            except (ValueError, OSError, cv2.error) as exc:
                failures.append(dict(manifest=name, path=str(path), reason=str(exc)))
            if (index + 1) % 100 == 0:
                print(f'{name}: processed {index + 1}/{len(rows)}', flush=True)
        target = root / (Path(name).stem + '_depth.csv')
        temporary = target.with_suffix('.tmp')
        with temporary.open('w', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            writer.writerows(output)
        temporary.replace(target)
        summary[name] = dict(input=len(rows), accepted=len(output),
                             genuine=sum(row['label'] == '1' for row in output),
                             recovered_by_fallback=sum(row['detection_threshold'] < 0.8 for row in output))
    saved.write_text(json.dumps(provenance, indent=2))
    (root / 'depth_failures.json').write_text(json.dumps(failures, indent=2))
    (root / 'depth_summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    if failures and not args.allow_failed:
        raise RuntimeError(f'{len(failures)} images need review. Inspect depth_failures.json. '
                           'After reviewing exclusions, rerun with --allow-failed to accept the logged subset.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--csv', nargs='+', default=['train.csv', 'val.csv', 'test.csv'])
    parser.add_argument('--namespace', required=True)
    parser.add_argument('--prnet-dir', type=Path, default=ROOT / 'checkpoints/prnet')
    parser.add_argument('--detector', type=Path, default=ROOT / 'checkpoints/face_detection_yunet_2026may.onnx')
    parser.add_argument('--limit-per-class', type=int)
    parser.add_argument('--allow-failed', action='store_true')
    arguments = parser.parse_args()
    if arguments.limit_per_class is not None and arguments.limit_per_class < 1:
        parser.error('--limit-per-class must be positive')
    cv2.setNumThreads(2)
    prepare(arguments)
