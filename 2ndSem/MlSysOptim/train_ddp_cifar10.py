#!/usr/bin/env python3
"""
============================================================
CIFAR-10 Distributed Training with PyTorch DDP
ML Systems Optimization Assignment — BITS Pilani WILP
============================================================

HOW TO RUN:
  pip install torch torchvision matplotlib

  # Run with 1 and 2 workers (CPU):
  python train_ddp_cifar10.py --workers 1 2 --epochs 20 --batch-size 128

  # Run with 1, 2, and 4 workers (CPU):
  python train_ddp_cifar10.py --workers 1 2 4 --epochs 20 --batch-size 128

  # If you have GPUs:
  python train_ddp_cifar10.py --workers 1 2 --epochs 20 --batch-size 128 --use-gpu

  Results are saved to ./results/
  Plots are saved to ./results/training_results.png
============================================================
"""

import os
import time
import json
import argparse
import numpy as np

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend (works without display)
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

import torchvision
import torchvision.transforms as transforms
import torchvision.models as models


# ============================================================
# SECTION 1: MODEL ARCHITECTURE
# ============================================================

class ResNet18CIFAR10(nn.Module):
    """
    ResNet-18 adapted for CIFAR-10 (32×32 pixel inputs).

    Why ResNet-18?
    - Proven architecture with ~11 million parameters.
    - Deep enough to learn complex features, not so large as to be impractical.
    - Well-documented baseline: achieves ~93% on CIFAR-10 with full training.

    Adaptations from standard ImageNet ResNet-18:
    1. First convolution: 7×7, stride 2 → 3×3, stride 1
       (Prevents excessive spatial downsampling on small 32×32 images)
    2. MaxPool removed (replaced by Identity layer)
       (32×32 is already small; early pooling destroys too much spatial info)
    3. Final FC layer: 1000 classes → 10 classes
    """
    def __init__(self, num_classes=10):
        super().__init__()
        base = models.resnet18(weights=None)
        # Adapt conv1 for CIFAR-10 small input
        base.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        # Remove maxpool (identity = do nothing)
        base.maxpool = nn.Identity()
        # Adapt final classification head
        base.fc = nn.Linear(512, num_classes)
        self.model = base

    def forward(self, x):
        return self.model(x)


# ============================================================
# SECTION 2: DATA LOADING
# ============================================================

def get_transforms():
    """
    Standard CIFAR-10 data augmentation pipeline.

    Training augmentations (reduce overfitting):
    - RandomCrop: Crop 32×32 with 4-pixel padding (shifts the image slightly)
    - RandomHorizontalFlip: Mirror image left-right with 50% probability
    - Normalize: Zero-mean, unit-variance using CIFAR-10 dataset statistics

    Test augmentations: Only normalization (no random ops during evaluation)
    """
    # CIFAR-10 channel means and standard deviations (pre-computed from dataset)
    mean = [0.4914, 0.4822, 0.4465]
    std  = [0.2023, 0.1994, 0.2010]

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    return train_transform, test_transform


def get_dataloaders(rank, world_size, batch_size, data_dir='./data'):
    """
    Create distributed data loaders.

    DistributedSampler: Ensures each worker (rank) receives a UNIQUE,
    non-overlapping subset of the training data. For example, with
    world_size=2 and 50,000 training images:
      - Worker 0 (rank=0) sees images 0, 2, 4, 6, ...  (25,000 images)
      - Worker 1 (rank=1) sees images 1, 3, 5, 7, ...  (25,000 images)
    This is the core of data parallelism.
    """
    train_transform, test_transform = get_transforms()

    train_dataset = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=train_transform
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=test_transform
    )

    # Distributed sampler — assigns non-overlapping subsets to each worker
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        drop_last=False
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,     # Use distributed sampler instead of shuffle=True
        num_workers=2,             # Parallel data loading within each process
        pin_memory=True,           # Faster CPU→GPU transfer (no-op on CPU)
        drop_last=True             # Drop last incomplete batch for stability
    )

    # All workers evaluate on the FULL test set; only rank 0 reports results
    test_loader = DataLoader(
        test_dataset,
        batch_size=256,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    return train_loader, test_loader, train_sampler


# ============================================================
# SECTION 3: COMMUNICATION OVERHEAD MEASUREMENT
# ============================================================

class CommTimer:
    """
    Measures gradient synchronization (all-reduce) overhead.

    Strategy: Compare backward time WITH synchronization (DDP normal mode)
    vs WITHOUT synchronization (model.no_sync() context), per epoch.
    Communication overhead = total backward time - compute-only backward time.
    """
    def __init__(self):
        self.compute_backward_time = 0.0  # Time WITHOUT sync
        self.total_backward_time = 0.0    # Time WITH sync
        self.n_steps = 0

    def reset(self):
        self.compute_backward_time = 0.0
        self.total_backward_time = 0.0
        self.n_steps = 0

    @property
    def communication_time(self):
        return max(0.0, self.total_backward_time - self.compute_backward_time)

    @property
    def comm_fraction(self):
        if self.total_backward_time == 0:
            return 0.0
        return self.communication_time / self.total_backward_time


# ============================================================
# SECTION 4: TRAINING LOOP
# ============================================================

def train_epoch(model, loader, sampler, optimizer, criterion, device, epoch, comm_timer):
    """
    Single training epoch with communication overhead measurement.
    """
    model.train()
    # CRITICAL: Set epoch on sampler so each epoch uses a different shuffle
    sampler.set_epoch(epoch)

    total_loss = 0.0
    correct = 0
    total_samples = 0

    comm_timer.reset()
    epoch_start = time.perf_counter()

    for batch_idx, (inputs, targets) in enumerate(loader):
        inputs  = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        # --- Forward pass ---
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # --- Measure backward WITH gradient synchronization (normal DDP) ---
        bwd_start = time.perf_counter()
        loss.backward()   # DDP automatically all-reduces gradients here
        bwd_with_sync = time.perf_counter() - bwd_start
        comm_timer.total_backward_time += bwd_with_sync

        # --- Measure backward WITHOUT synchronization (compute-only baseline) ---
        # We do this on every ~10th step to avoid doubling training time
        if batch_idx % 10 == 0:
            # Recompute to get gradients without sync overhead
            with model.no_sync():
                outputs_nosync = model(inputs)
                loss_nosync = criterion(outputs_nosync, targets)
                bwd_start_nosync = time.perf_counter()
                loss_nosync.backward()
                bwd_nosync = time.perf_counter() - bwd_start_nosync
                comm_timer.compute_backward_time += bwd_nosync
                comm_timer.n_steps += 1

        optimizer.step()

        # --- Track accuracy and loss ---
        total_loss    += loss.item()
        _, predicted   = outputs.max(1)
        total_samples += targets.size(0)
        correct       += predicted.eq(targets).sum().item()

    epoch_time = time.perf_counter() - epoch_start
    train_loss = total_loss / len(loader)
    train_acc  = 100.0 * correct / total_samples

    return train_loss, train_acc, epoch_time


def evaluate(model, loader, criterion, device):
    """Evaluate model on the full test set."""
    model.eval()
    total_loss = 0.0
    correct    = 0
    total      = 0

    with torch.no_grad():
        for inputs, targets in loader:
            inputs  = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            outputs = model(inputs)
            loss    = criterion(outputs, targets)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total    += targets.size(0)
            correct  += predicted.eq(targets).sum().item()

    return total_loss / len(loader), 100.0 * correct / total


# ============================================================
# SECTION 5: MAIN WORKER (one per process)
# ============================================================

def worker(rank, world_size, args):
    """
    This function runs inside each spawned process.
    rank=0 is the "master" worker that prints and saves results.
    """

    # ---- Set up distributed process group ----
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # Backend selection:
    #   NCCL — highly optimized for NVIDIA GPUs (collective ops via NVLink/PCIe)
    #   Gloo — CPU-compatible, cross-platform, slower but universally available
    use_gpu = args.use_gpu and torch.cuda.is_available()
    backend = 'nccl' if use_gpu else 'gloo'

    dist.init_process_group(
        backend=backend,
        rank=rank,
        world_size=world_size
    )

    # ---- Device setup ----
    if use_gpu:
        device = torch.device(f'cuda:{rank}')
        torch.cuda.set_device(device)
    else:
        device = torch.device('cpu')

    torch.manual_seed(42 + rank)  # Reproducibility

    if rank == 0:
        print(f"\n{'='*62}")
        print(f"  Configuration: {world_size} Worker(s)")
        print(f"{'='*62}")
        print(f"  Backend          : {backend.upper()}")
        print(f"  Device           : {device}")
        print(f"  Batch/worker     : {args.batch_size}")
        print(f"  Effective batch  : {args.batch_size * world_size}")
        print(f"  Epochs           : {args.epochs}")
        print(f"  LR (scaled)      : {0.05 * world_size:.4f}")
        print(f"{'='*62}\n")

    # ---- Model ----
    model = ResNet18CIFAR10(num_classes=10).to(device)

    # Wrap model in DDP:
    # DDP hooks into backward() to call all_reduce on gradients automatically.
    # After all_reduce, every worker has the average gradient → same parameter update.
    ddp_model = DDP(
        model,
        device_ids=[rank] if use_gpu else None,
        find_unused_parameters=False
    )

    # ---- Data ----
    train_loader, test_loader, train_sampler = get_dataloaders(
        rank, world_size, args.batch_size, args.data_dir
    )

    # ---- Optimizer ----
    # Linear Scaling Rule (Goyal et al., 2017):
    # When effective batch size scales with N workers, scale LR proportionally.
    # This preserves the "effective" learning rate per sample.
    base_lr = 0.05
    lr = base_lr * world_size

    optimizer = optim.SGD(
        ddp_model.parameters(),
        lr=lr,
        momentum=0.9,
        weight_decay=5e-4,
        nesterov=True
    )

    criterion = nn.CrossEntropyLoss()

    # Cosine annealing: smoothly decays LR from lr → 0 over all epochs
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

    # ---- Communication Timer ----
    comm_timer = CommTimer()

    # ---- Results storage ----
    results = {
        'world_size'          : world_size,
        'backend'             : backend,
        'epochs'              : args.epochs,
        'batch_size_per_worker': args.batch_size,
        'effective_batch_size': args.batch_size * world_size,
        'epoch_times'         : [],
        'train_losses'        : [],
        'train_accs'          : [],
        'test_losses'         : [],
        'test_accs'           : [],
        'comm_times'          : [],
        'comm_fractions'      : [],
        'total_training_time' : 0.0,
    }

    training_start = time.perf_counter()

    # ---- Training loop ----
    for epoch in range(args.epochs):

        train_loss, train_acc, epoch_time = train_epoch(
            ddp_model, train_loader, train_sampler,
            optimizer, criterion, device, epoch, comm_timer
        )

        # Only rank 0 evaluates (avoids redundant work)
        test_loss, test_acc = evaluate(ddp_model, test_loader, criterion, device)

        scheduler.step()

        # Compute communication fraction this epoch
        comm_frac = comm_timer.comm_fraction * 100  # as percentage

        results['epoch_times'].append(round(epoch_time, 4))
        results['train_losses'].append(round(train_loss, 6))
        results['train_accs'].append(round(train_acc, 4))
        results['test_losses'].append(round(test_loss, 6))
        results['test_accs'].append(round(test_acc, 4))
        results['comm_fractions'].append(round(comm_frac, 2))

        if rank == 0:
            curr_lr = scheduler.get_last_lr()[0]
            print(
                f"  Epoch [{epoch+1:3d}/{args.epochs}] | "
                f"Time: {epoch_time:6.2f}s | "
                f"Loss: {train_loss:.4f} | "
                f"Train: {train_acc:6.2f}% | "
                f"Test: {test_acc:6.2f}% | "
                f"Comm: {comm_frac:5.1f}% | "
                f"LR: {curr_lr:.5f}"
            )

    total_time = time.perf_counter() - training_start
    results['total_training_time'] = round(total_time, 2)

    # ---- Save results (rank 0 only) ----
    if rank == 0:
        print(f"\n  Total Training Time  : {total_time:.2f}s")
        print(f"  Final Test Accuracy  : {results['test_accs'][-1]:.2f}%")
        print(f"  Avg Epoch Time       : {np.mean(results['epoch_times']):.2f}s")
        print(f"  Avg Comm Overhead    : {np.mean(results['comm_fractions']):.1f}%")

        os.makedirs(args.output_dir, exist_ok=True)
        out_file = os.path.join(args.output_dir, f'results_workers_{world_size}.json')
        with open(out_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"  Results saved to: {out_file}")

    dist.destroy_process_group()


# ============================================================
# SECTION 6: ANALYSIS AND PLOTTING
# ============================================================

def compute_speedup_metrics(results_dict):
    """
    Compute all performance metrics from raw results.

    Speedup S(N)     = T1 / TN
    Efficiency E(N)  = S(N) / N × 100%
    Accuracy Gap     = Acc(1) - Acc(N)
    """
    baseline_time = results_dict[1]['total_training_time']
    baseline_acc  = results_dict[1]['test_accs'][-1]
    metrics = {}

    for n_workers, res in sorted(results_dict.items()):
        t_n      = res['total_training_time']
        speedup  = baseline_time / t_n
        eff      = (speedup / n_workers) * 100.0
        acc_gap  = baseline_acc - res['test_accs'][-1]
        avg_epoch_time = np.mean(res['epoch_times'])
        avg_comm_frac  = np.mean(res['comm_fractions'])

        metrics[n_workers] = {
            'total_time'    : t_n,
            'speedup'       : round(speedup, 3),
            'efficiency'    : round(eff, 2),
            'acc_gap'       : round(acc_gap, 2),
            'avg_epoch_time': round(avg_epoch_time, 2),
            'final_test_acc': res['test_accs'][-1],
            'avg_comm_frac' : round(avg_comm_frac, 2),
        }

    return metrics


def plot_all_results(results_dict, output_dir):
    """Generate 4-panel performance plot."""
    os.makedirs(output_dir, exist_ok=True)
    metrics    = compute_speedup_metrics(results_dict)
    n_list     = sorted(results_dict.keys())
    palette    = ['#1f77b4', '#2ca02c', '#d62728', '#9467bd']

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        'CIFAR-10 Distributed Training: Performance Analysis\n'
        'ResNet-18 | PyTorch DDP | Data Parallelism',
        fontsize=13, fontweight='bold', y=1.01
    )

    # --- Plot 1: Convergence curves (Test Accuracy vs Epoch) ---
    ax = axes[0, 0]
    for i, n in enumerate(n_list):
        epochs = list(range(1, len(results_dict[n]['test_accs']) + 1))
        ax.plot(epochs, results_dict[n]['test_accs'],
                label=f'{n} Worker(s)', color=palette[i], linewidth=2, marker='o',
                markersize=3)
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Test Accuracy (%)', fontsize=11)
    ax.set_title('Convergence: Test Accuracy vs Epoch', fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # --- Plot 2: Speedup vs Workers ---
    ax = axes[0, 1]
    speedups      = [metrics[n]['speedup'] for n in n_list]
    ideal_speedup = n_list
    ax.plot(n_list, speedups, 'bo-', linewidth=2.5, markersize=9,
            label='Actual Speedup S(N)', zorder=5)
    ax.plot(n_list, ideal_speedup, 'r--', linewidth=2,
            label='Ideal Linear Speedup', alpha=0.7)
    for n, s in zip(n_list, speedups):
        ax.annotate(f'{s:.2f}x', (n, s), textcoords="offset points",
                    xytext=(0, 10), ha='center', fontsize=10, fontweight='bold')
    ax.set_xlabel('Number of Workers N', fontsize=11)
    ax.set_ylabel('Speedup S(N) = T₁ / Tₙ', fontsize=11)
    ax.set_title('Speedup vs Number of Workers', fontweight='bold')
    ax.set_xticks(n_list)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # --- Plot 3: Parallel Efficiency ---
    ax = axes[1, 0]
    efficiencies = [metrics[n]['efficiency'] for n in n_list]
    bars = ax.bar(n_list, efficiencies, color=[palette[i] for i in range(len(n_list))],
                  alpha=0.85, edgecolor='black', linewidth=0.8, width=0.5)
    ax.axhline(y=100, color='red', linestyle='--', linewidth=1.5,
               label='100% (Ideal Efficiency)')
    for bar, eff in zip(bars, efficiencies):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1.5,
                f'{eff:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax.set_xlabel('Number of Workers N', fontsize=11)
    ax.set_ylabel('Parallel Efficiency E(N) = S(N)/N × 100%', fontsize=11)
    ax.set_title('Parallel Efficiency vs Workers', fontweight='bold')
    ax.set_xticks(n_list)
    ax.set_ylim(0, 115)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    # --- Plot 4: Average Epoch Time & Communication Overhead ---
    ax = axes[1, 1]
    avg_epoch_times = [metrics[n]['avg_epoch_time'] for n in n_list]
    comm_fracs      = [metrics[n]['avg_comm_frac'] for n in n_list]

    x = np.arange(len(n_list))
    width = 0.35
    bars1 = ax.bar(x - width/2, avg_epoch_times, width, label='Avg Epoch Time (s)',
                   color='steelblue', alpha=0.85, edgecolor='black', linewidth=0.8)

    ax2 = ax.twinx()
    bars2 = ax2.bar(x + width/2, comm_fracs, width, label='Comm Overhead (%)',
                    color='tomato', alpha=0.85, edgecolor='black', linewidth=0.8)

    ax.set_xlabel('Number of Workers N', fontsize=11)
    ax.set_ylabel('Average Epoch Time (s)', fontsize=11, color='steelblue')
    ax2.set_ylabel('Communication Overhead (%)', fontsize=11, color='tomato')
    ax.set_title('Epoch Time & Communication Overhead', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{n} Workers' for n in n_list])

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=10, loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'training_results.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  [PLOTS] Saved to: {plot_path}")
    return plot_path


def print_summary_table(results_dict):
    """Print formatted metrics table to console."""
    metrics = compute_speedup_metrics(results_dict)

    print("\n" + "="*90)
    print(f"{'PERFORMANCE METRICS SUMMARY':^90}")
    print("="*90)
    header = (f"{'Workers':^10} | {'Total Time(s)':^14} | {'Epoch Time(s)':^13} | "
              f"{'Speedup S(N)':^13} | {'Efficiency(%)':^13} | {'Comm(%)':^8} | {'Test Acc(%)':^11}")
    print(header)
    print("-"*90)
    for n in sorted(metrics.keys()):
        m = metrics[n]
        row = (f"{n:^10} | {m['total_time']:^14.2f} | {m['avg_epoch_time']:^13.2f} | "
               f"{m['speedup']:^13.3f} | {m['efficiency']:^13.2f} | "
               f"{m['avg_comm_frac']:^8.1f} | {m['final_test_acc']:^11.2f}")
        print(row)
    print("="*90)

    # Accuracy gap table
    baseline_acc = metrics[1]['final_test_acc']
    print(f"\n  Accuracy Gap (vs 1 worker, baseline={baseline_acc:.2f}%):")
    for n in sorted(metrics.keys()):
        gap = metrics[n]['acc_gap']
        sign = '+' if gap > 0 else ''
        print(f"    {n} workers: {sign}{gap:.2f}% accuracy gap")
    print()


# ============================================================
# SECTION 7: ENTRY POINT
# ============================================================

def run_experiment(world_size, args):
    """Spawn `world_size` processes, each running worker()."""
    mp.spawn(
        worker,
        args=(world_size, args),
        nprocs=world_size,
        join=True
    )


def main():
    parser = argparse.ArgumentParser(
        description='CIFAR-10 Distributed Training Benchmark (PyTorch DDP)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--epochs',     type=int,   default=20,
                        help='Training epochs per configuration')
    parser.add_argument('--batch-size', type=int,   default=128,
                        help='Mini-batch size PER worker')
    parser.add_argument('--data-dir',   type=str,   default='./data',
                        help='Where to download CIFAR-10')
    parser.add_argument('--output-dir', type=str,   default='./results',
                        help='Where to save results and plots')
    parser.add_argument('--use-gpu',    action='store_true',
                        help='Use CUDA GPUs if available')
    parser.add_argument('--workers',    type=int, nargs='+', default=[1, 2],
                        help='Worker counts to benchmark, e.g. --workers 1 2 4')
    args = parser.parse_args()

    print("\n" + "="*62)
    print("  CIFAR-10 Distributed Training Benchmark")
    print("  ML Systems Optimization — BITS Pilani WILP")
    print("="*62)
    print(f"  Configurations to run : {args.workers}")
    print(f"  Epochs each           : {args.epochs}")
    print(f"  Batch size / worker   : {args.batch_size}")
    print(f"  GPU mode              : {'ON' if args.use_gpu else 'OFF (CPU)'}")
    print("="*62)

    all_results = {}

    for n_workers in args.workers:
        print(f"\n{'='*62}")
        print(f"  EXPERIMENT: {n_workers} Worker(s)")
        print(f"{'='*62}")
        run_experiment(n_workers, args)

        # Load results saved by rank 0
        result_file = os.path.join(args.output_dir, f'results_workers_{n_workers}.json')
        if os.path.exists(result_file):
            with open(result_file, 'r') as f:
                all_results[n_workers] = json.load(f)
        else:
            print(f"  WARNING: Result file not found for {n_workers} workers.")

    # Final analysis
    if len(all_results) >= 2:
        print_summary_table(all_results)
        plot_all_results(all_results, args.output_dir)

        # Save combined results
        combined = os.path.join(args.output_dir, 'combined_metrics.json')
        metrics  = compute_speedup_metrics(all_results)
        with open(combined, 'w') as f:
            json.dump({str(k): v for k, v in metrics.items()}, f, indent=2)
        print(f"  Combined metrics saved to: {combined}")
    else:
        print("  NOTE: Run at least 2 configurations to generate comparison plots.")

    print("\n  Done! Check ./results/ for all output files.")


if __name__ == '__main__':
    main()
