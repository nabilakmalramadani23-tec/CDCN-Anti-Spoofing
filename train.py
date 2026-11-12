import argparse
from collections import Counter
import random

import torch
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from datasets.FASDataset import FASDataset, check_disjoint
from models.loss import DepthLoss
from trainer.FASTrainer import FASTrainer
from utils.eval import evaluation_transform
from utils.transform import RandomGammaCorrection
from utils.utils import build_network, get_device, get_optimizer, read_cfg


def main():
    parser = argparse.ArgumentParser(description='Train CDCN or CDCNpp')
    parser.add_argument('--config', default='config/CDCN_depth.yaml')
    parser.add_argument('--init-checkpoint', help='Initialize model weights from a trained checkpoint')
    parser.add_argument('--resume', help='Resume a PRNet-depth run from latest.pth')
    parser.add_argument('--check-data', action='store_true', help='Validate all depth maps and split separation')
    parser.add_argument('--allow-legacy-binary', action='store_true', help='Explicitly reproduce historical binary-map experiments')
    args = parser.parse_args()
    cfg = read_cfg(args.config)
    if cfg['dataset'].get('supervision') == 'prnet_depth':
        from trainer.depth_training import train_depth
        train_depth(cfg, args)
        return
    if not args.allow_legacy_binary or args.resume or args.check_data:
        parser.error('Use config/CDCN_depth.yaml for PRNet-depth CDCN. '
                     'Historical binary-map training requires --allow-legacy-binary.')
    if 'seed' in cfg['train']:
        random.seed(cfg['train']['seed'])
        torch.manual_seed(cfg['train']['seed'])
    device = get_device(cfg)
    dataset = cfg['dataset']
    input_size = cfg['model']['input_size']
    min_gamma, max_gamma = dataset['augmentation']['gamma_correction']
    train_transform = transforms.Compose([
        RandomGammaCorrection(max_gamma=max_gamma, min_gamma=min_gamma),
        transforms.RandomResizedCrop(input_size[0]),
        transforms.RandomRotation(dataset['augmentation']['rotation_range']),
        transforms.RandomHorizontalFlip(),
        evaluation_transform(cfg)
    ])
    dataset_options = dict(
        root_dir=dataset['root'],
        depth_map_size=cfg['model']['depth_map_size'],
        smoothing=cfg['train']['smoothing']
    )
    trainset = FASDataset(csv_file=dataset['train_set'], transform=train_transform, **dataset_options)
    valset = FASDataset(csv_file=dataset['val_set'], transform=evaluation_transform(cfg), **dataset_options)
    check_disjoint(trainset.samples, valset.samples, 'Train', 'Validation')
    sampler = None
    if cfg['train'].get('balance_classes', False):
        counts = Counter(label for _, label in trainset.samples)
        weights = [1 / counts[label] for _, label in trainset.samples]
        samples_per_epoch = cfg['train'].get('samples_per_epoch', len(trainset))
        sampler = torch.utils.data.WeightedRandomSampler(weights, samples_per_epoch)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=cfg['train']['batch_size'], shuffle=sampler is None, sampler=sampler
    )
    valloader = torch.utils.data.DataLoader(valset, batch_size=cfg['val']['batch_size'])

    network = build_network(cfg).to(device)
    if args.init_checkpoint:
        checkpoint = torch.load(args.init_checkpoint, map_location='cpu', weights_only=True)
        network.load_state_dict(checkpoint['state_dict'])
        print(f'Initialized model weights from {args.init_checkpoint}')
    optimizer = get_optimizer(cfg, network)
    lr_scheduler = StepLR(optimizer=optimizer, step_size=30, gamma=0.1)
    print(f'Training on {device} with {len(trainset)} train and {len(valset)} validation images')
    with SummaryWriter(cfg['log_dir']) as writer:
        trainer = FASTrainer(cfg, network, optimizer, DepthLoss(device), lr_scheduler, device,
                             trainloader, valloader, writer)
        trainer.train()


if __name__ == '__main__':
    main()
