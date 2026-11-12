import argparse
import json
import math
from pathlib import Path

import torch

from datasets.FASDataset import FASDataset, check_disjoint, read_samples
from models.loss import DepthLoss
from utils.eval import evaluation_transform, predict
from utils.utils import build_network, get_device, read_cfg


def main():
    parser = argparse.ArgumentParser(description='Evaluate a CDCN checkpoint on labeled images')
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', help='Checkpoint path, defaults to checkpoint in the YAML')
    split = parser.add_mutually_exclusive_group(required=True)
    split.add_argument('--test-csv', help='CSV path relative to the data root or an absolute path')
    split.add_argument('--validation', action='store_true', help='Evaluate the configured validation split')
    parser.add_argument('--data-root', help='Directory for test image paths, defaults to dataset.root')
    parser.add_argument('--threshold', type=float, help='Defaults to val.threshold in the YAML or 0.5')
    args = parser.parse_args()
    cfg = read_cfg(args.config)
    if cfg['dataset'].get('supervision') == 'prnet_depth':
        from trainer.depth_training import evaluate_depth
        evaluate_depth(cfg, args)
        return
    threshold = args.threshold if args.threshold is not None else float(cfg['val'].get('threshold', 0.5))
    checkpoint_path = args.checkpoint or cfg.get('checkpoint')
    if not checkpoint_path:
        parser.error('--checkpoint is required when the YAML has no checkpoint')
    if not math.isfinite(threshold) or threshold < 0:
        parser.error('--threshold must be finite and nonnegative')
    if args.validation and args.data_root:
        parser.error('--data-root applies only to --test-csv')

    device = get_device(cfg)
    dataset = cfg['dataset']
    samples = FASDataset(
        root_dir=args.data_root or dataset['root'],
        csv_file=dataset['val_set'] if args.validation else args.test_csv,
        depth_map_size=cfg['model']['depth_map_size'],
        transform=evaluation_transform(cfg),
        smoothing=cfg['train']['smoothing']
    )
    split_name = 'Validation' if args.validation else 'Test'
    excluded_splits = ('train_set',) if args.validation else ('train_set', 'val_set')
    for split in excluded_splits:
        manifest = Path(dataset['root']) / dataset[split]
        if manifest.is_file():
            check_disjoint(read_samples(dataset['root'], dataset[split]), samples.samples, split, split_name)
    loader = torch.utils.data.DataLoader(samples, batch_size=cfg['val']['batch_size'])
    network = build_network(cfg).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    network.load_state_dict(checkpoint['state_dict'])
    network.eval()
    criterion = DepthLoss(device)
    loss_sum = 0.0
    correct = 0
    attacks = genuine = attacks_accepted = genuine_rejected = 0

    with torch.inference_mode():
        for images, targets, labels in loader:
            depth_maps = network(images.to(device))[0]
            loss_sum += criterion(depth_maps, targets.to(device)).item() * images.size(0)
            predictions, _ = predict(depth_maps, threshold)
            labels = labels.flatten().to(device)
            correct += (predictions == labels).sum().item()
            is_attack = labels == 0
            attacks += is_attack.sum().item()
            genuine += (labels == 1).sum().item()
            attacks_accepted += (is_attack & (predictions == 1)).sum().item()
            genuine_rejected += ((labels == 1) & (predictions == 0)).sum().item()

    apcer = attacks_accepted / attacks
    bpcer = genuine_rejected / genuine
    print(json.dumps({
        'threshold': threshold,
        'samples': len(samples),
        'attack_samples': attacks,
        'genuine_samples': genuine,
        'attacks_accepted': attacks_accepted,
        'genuine_rejected': genuine_rejected,
        'loss': loss_sum / len(samples),
        'accuracy_pct': 100 * correct / len(samples),
        'apcer_pct': 100 * apcer,
        'bpcer_pct': 100 * bpcer,
        'acer_pct': 100 * (apcer + bpcer) / 2
    }, indent=2))


if __name__ == '__main__':
    main()
