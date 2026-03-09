# --- Dependencies and Setup ---
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import io
import sys
import subprocess
import time

# Check and install required packages
try:
    from tqdm import tqdm
except ImportError:
    print("tqdm not found. Installing...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "tqdm"])
    from tqdm import tqdm

try:
    import pandas as pd
except ImportError:
    print("Pandas not found. Installing...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pandas"])
    import pandas as pd
    
try:
    import colorama
    colorama.init()
    class C:
        HEADER = '\033[95m'; BLUE = '\033[94m'; CYAN = '\033[96m'; GREEN = '\033[92m'
        YELLOW = '\033[93m'; RED = '\033[91m'; END = '\033[0m'; BOLD = '\033[1m'; UNDERLINE = '\033[4m'
except ImportError:
    class C:
        HEADER = BOLD = BLUE = CYAN = GREEN = YELLOW = RED = END = ''


# --- Configuration ---
CONFIG = {
    "epochs": 5,
    "batch_size": 256, # Increased batch size due to performance improvements
    "lr_main": 1e-3,
    "lr_policy": 5e-4, 
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "lambda_bits": 0.001, 
    "bit_choices": torch.tensor([2, 4, 8], dtype=torch.float32), # Use a tensor for easier ops
    "gumbel_temp_initial": 5.0,
    "gumbel_temp_final": 0.5,
    "output_filename": "comparison_report.png"
}
CONFIG['bit_choices'] = CONFIG['bit_choices'].to(CONFIG['device'])

# --- Data Loading (Unchanged) ---
def get_data_loaders(batch_size, shuffle_train=True):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle_train, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    return train_loader, test_loader

# --- Naive FFN Model (Unchanged) ---
class NaiveFFN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(784, 256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 10))
    def forward(self, x):
        return self.layers(x.view(x.size(0), -1))

# --- IAQ Model Components (QuantizedLayer is now BATCH-OPTIMIZED) ---
class FakeQuantize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, bits):
        if bits == 32: return x
        q_level = 2**bits - 1
        min_val, max_val = x.min(), x.max()
        scale = (max_val - min_val) / (q_level + 1e-9)
        zero_point = min_val
        quantized = torch.round((x - zero_point) / (scale + 1e-9))
        dequantized = quantized * scale + zero_point
        return dequantized
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None
apply_fake_quant = FakeQuantize.apply

class QuantizationController(nn.Module):
    def __init__(self, num_choices):
        super().__init__()
        self.controller_net = nn.Sequential(nn.Linear(3, 16), nn.ReLU(), nn.Linear(16, num_choices))
    def forward(self, x):
        return self.controller_net(x)

class QuantizedLayer(nn.Module):
    def __init__(self, in_features, out_features, bit_choices):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.relu = nn.ReLU()
        self.bit_choices_values = bit_choices
        self.controller = QuantizationController(len(bit_choices))

    def forward(self, x, temp, is_eval=False):
        batch_size = x.size(0)
        
        # 1. Calculate activation statistics (BATCHED)
        with torch.no_grad():
            act_mean = x.mean(dim=1)
            act_var = x.var(dim=1)
            act_sparsity = (x == 0).float().mean(dim=1)
            stats = torch.stack([act_mean, act_var, act_sparsity], dim=1) # Shape: [batch_size, 3]

        # 2. Controller decides on bitwidth (BATCHED)
        logits = self.controller(stats) # Shape: [batch_size, num_choices]

        # 3. Sample bitwidth (BATCHED)
        if is_eval:
            choice_indices = logits.argmax(dim=1) # Shape: [batch_size]
            log_probs = torch.zeros(batch_size, device=x.device) # Not needed
        else:
            gumbel_probs = F.gumbel_softmax(logits, tau=temp, hard=True) # Shape: [batch_size, num_choices]
            choice_indices = gumbel_probs.argmax(dim=1)
            # Gather the log_prob of the chosen action
            log_probs = torch.log(F.softmax(logits, dim=-1).gather(1, choice_indices.unsqueeze(1)).squeeze() + 1e-9)
        
        chosen_bits = self.bit_choices_values[choice_indices] # Shape: [batch_size]
        
        # 4. Forward pass with grouped quantization (THE KEY OPTIMIZATION)
        output = torch.zeros(batch_size, self.linear.out_features, device=x.device)
        
        # Group samples by chosen bitwidth
        for bit_val in chosen_bits.unique():
            # Find which samples in the batch use this bitwidth
            mask = (chosen_bits == bit_val)
            
            # Quantize weight ONCE for this group
            quantized_weight = apply_fake_quant(self.linear.weight, bit_val)
            
            # Apply linear layer to the subset of inputs
            output[mask] = F.linear(x[mask], quantized_weight, self.linear.bias)
        
        output = self.relu(output)
        
        return output, log_probs, chosen_bits

class IAQ_FFN(nn.Module):
    def __init__(self, bit_choices):
        super().__init__()
        self.quant_layer1 = QuantizedLayer(784, 256, bit_choices)
        self.quant_layer2 = QuantizedLayer(256, 128, bit_choices)
        self.quant_layer3 = QuantizedLayer(128, 64, bit_choices)
        self.quant_layer4 = QuantizedLayer(64, 32, bit_choices)
        self.output_layer = nn.Linear(32, 10)
        self.layers = [self.quant_layer1, self.quant_layer2, self.quant_layer3, self.quant_layer4]

    def forward(self, x, temp=1.0, is_eval=False):
        x = x.view(x.size(0), -1)
        batch_log_probs = []
        batch_bits = []
        
        for layer in self.layers:
            x, log_p, bits = layer(x, temp, is_eval)
            batch_log_probs.append(log_p)
            batch_bits.append(bits)

        # Transpose lists of tensors to get per-sample totals
        log_probs_per_sample = torch.stack(batch_log_probs, dim=1).sum(dim=1)
        bits_per_sample = torch.stack(batch_bits, dim=1)
        
        output = self.output_layer(x)
        return output, log_probs_per_sample, bits_per_sample

# --- Training and Evaluation Loops (IAQ loops now BATCHED) ---

def train_naive(model, device, train_loader, optimizer, epoch, stats):
    model.train()
    pbar = tqdm(train_loader, desc=f"{C.CYAN}Epoch {epoch} [Train]{C.END}", unit="batch")
    for data, target in pbar:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad(); output = model(data); loss = F.cross_entropy(output, target)
        loss.backward(); optimizer.step(); pbar.set_postfix(loss=f"{loss.item():.4f}")
    stats['train_loss'].append(loss.item())

def test_naive(model, device, test_loader, stats):
    model.eval(); test_loss = 0; correct = 0
    pbar = tqdm(test_loader, desc=f"{C.YELLOW}Epoch {epoch} [Test]{C.END}", unit="batch")
    with torch.no_grad():
        for data, target in pbar:
            data, target = data.to(device), target.to(device)
            output = model(data); test_loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True); correct += pred.eq(target.view_as(pred)).sum().item()
            pbar.set_postfix(acc=f"{100. * correct / len(test_loader.dataset):.2f}%")
    test_loss /= len(test_loader.dataset); accuracy = 100. * correct / len(test_loader.dataset)
    stats['test_loss'].append(test_loss); stats['test_acc'].append(accuracy)
    print(f"  {C.YELLOW}└> Test Results: Avg loss: {test_loss:.4f}, Accuracy: {accuracy:.2f}%{C.END}")

def train_iaq(model, device, train_loader, optim_main, optim_policy, epoch, stats, temp_schedule):
    model.train()
    pbar = tqdm(train_loader, desc=f"{C.CYAN}Epoch {epoch} [Train]{C.END}", unit="batch")
    
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        global_step = (epoch - 1) * len(train_loader) + batch_idx
        temp = temp_schedule[global_step]

        optim_main.zero_grad(); optim_policy.zero_grad()
        
        output, log_probs, bits_per_sample = model(data, temp=temp)
        
        task_loss = F.cross_entropy(output, target)
        
        total_bits_per_sample = bits_per_sample.sum(dim=1)
        bit_cost = CONFIG['lambda_bits'] * total_bits_per_sample
        
        # Reward is calculated per-sample
        per_sample_task_loss = F.cross_entropy(output, target, reduction='none')
        reward = -per_sample_task_loss.detach() - bit_cost
        
        policy_loss = (-log_probs * reward).mean() # REINFORCE with baseline (mean reward) implicitly
        
        total_loss = task_loss + policy_loss
        total_loss.backward()
        
        optim_main.step(); optim_policy.step()

        stats['rewards'].append(reward.mean().item())
        stats['avg_bits'].append(total_bits_per_sample.mean().item())
        stats['train_loss'].append(task_loss.item())
        pbar.set_postfix(task_loss=f"{task_loss.item():.3f}", reward=f"{reward.mean().item():.3f}", avg_bits=f"{total_bits_per_sample.mean().item():.1f}", temp=f"{temp:.2f}")

def test_iaq(model, device, test_loader, epoch, stats):
    model.eval(); test_loss = 0; correct = 0
    pbar = tqdm(test_loader, desc=f"{C.YELLOW}Epoch {epoch} [Test]{C.END}", unit="batch")
    
    all_bits, all_losses, all_bit_decisions, all_correct, all_entropies = [], [], defaultdict(list), [], []

    with torch.no_grad():
        for data, target in pbar:
            data, target = data.to(device), target.to(device)
            output, _, bits_per_sample = model(data, is_eval=True)
            
            per_sample_loss = F.cross_entropy(output, target, reduction='none')
            test_loss += per_sample_loss.sum().item()
            pred = output.argmax(dim=1)
            
            is_correct = pred.eq(target)
            correct += is_correct.sum().item()
            
            total_bits_per_sample = bits_per_sample.sum(dim=1)
            all_bits.extend(total_bits_per_sample.cpu().tolist())
            all_losses.extend(per_sample_loss.cpu().tolist())
            all_correct.extend(is_correct.cpu().tolist())

            # Log controller entropy
            for layer in model.layers:
                stats_ = torch.stack([data.mean(dim=(1,2,3)), data.var(dim=(1,2,3)), (data==0).float().mean(dim=(1,2,3))], dim=1)
                logits = layer.controller(stats_)
                entropy = -(F.softmax(logits, dim=1) * F.log_softmax(logits, dim=1)).sum(dim=1)
                all_entropies.extend(entropy.cpu().tolist())

            for i in range(bits_per_sample.shape[1]):
                all_bit_decisions[i].extend(bits_per_sample[:, i].cpu().tolist())
            
            pbar.set_postfix(acc=f"{100. * correct / len(test_loader.dataset):.2f}%", avg_bits=f"{np.mean(all_bits):.2f}")

    test_loss /= len(test_loader.dataset); accuracy = 100. * correct / len(test_loader.dataset)
    avg_total_bits = np.mean(all_bits)

    stats['test_loss'].append(test_loss); stats['test_acc'].append(accuracy)
    stats['final_test_bits_vs_loss'] = (all_losses, all_bits)
    stats['final_bit_decisions'] = all_bit_decisions
    stats['final_bits_vs_correct'] = (all_bits, all_correct)
    stats['final_controller_entropies'] = all_entropies
    print(f"  {C.YELLOW}└> Test Results: Avg loss: {test_loss:.4f}, Accuracy: {accuracy:.2f}%, Avg Bits: {avg_total_bits:.2f}{C.END}")

# --- NEW: Statistics and Cost Analysis ---
def calculate_model_stats(model, model_type, iaq_stats=None):
    device = CONFIG['device']
    stats = {}
    
    # --- Parameter Count ---
    if model_type == 'naive':
        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        stats['params'] = params
        stats['params_backbone'] = params
        stats['params_controllers'] = 0
    else: # IAQ
        params_backbone = sum(p.numel() for n, p in model.named_parameters() if 'controller' not in n)
        params_controllers = sum(p.numel() for n, p in model.named_parameters() if 'controller' in n)
        stats['params'] = params_backbone + params_controllers
        stats['params_backbone'] = params_backbone
        stats['params_controllers'] = params_controllers

    # --- Model Size (MB) ---
    if model_type == 'naive':
        stats['size_mb'] = stats['params'] * 4 / (1024**2) # FP32
    else:
        avg_bits_per_weight = np.mean(list(iaq_stats['final_bit_decisions'].values()))
        backbone_size_bits = stats['params_backbone'] * avg_bits_per_weight
        controller_size_bits = stats['params_controllers'] * 32 # Controllers are FP32
        stats['size_mb'] = (backbone_size_bits + controller_size_bits) / (8 * 1024**2)

    # --- GFLOPs & Latency ---
    dummy_input = torch.randn(1, 784).to(device)
    total_flops = 0
    
    if model_type == 'naive':
        for layer in model.layers:
            if isinstance(layer, nn.Linear):
                total_flops += 2 * layer.in_features * layer.out_features
    else: # IAQ
        avg_bit_choices = [np.mean(v) for v in iaq_stats['final_bit_decisions'].values()]
        for i, layer in enumerate(model.layers):
            # FLOPs are independent of bit-depth in this simulation
            total_flops += 2 * layer.linear.in_features * layer.linear.out_features
    stats['gflops'] = total_flops / 1e9

    # Latency
    model.eval()
    with torch.no_grad():
        # Warmup
        for _ in range(10):
            _ = model(dummy_input.expand(128, -1)) if model_type == 'naive' else model(dummy_input.expand(128, -1), is_eval=True)
        
        torch.cuda.synchronize()
        start_time = time.time()
        for _ in range(100):
            _ = model(dummy_input.expand(128, -1)) if model_type == 'naive' else model(dummy_input.expand(128, -1), is_eval=True)
        torch.cuda.synchronize()
        end_time = time.time()
    
    stats['latency_ms'] = ((end_time - start_time) / (100 * 128)) * 1000 # Per-sample latency

    return stats

# --- Visualization (Now 4x2 grid) ---
def create_visualizations(naive_stats, iaq_stats):
    fig, axes = plt.subplots(4, 2, figsize=(16, 26))
    fig.suptitle('Naive FFN vs. Input-Adaptive Quantization (IAQ) FFN', fontsize=20, weight='bold')

    # Row 1: Basic Performance
    axes[0,0].plot(naive_stats['test_acc'], label=f"Naive (Final: {naive_stats['test_acc'][-1]:.2f}%)", marker='o'); axes[0,0].plot(iaq_stats['test_acc'], label=f"IAQ (Final: {iaq_stats['test_acc'][-1]:.2f}%)", marker='x'); axes[0,0].set_title("Test Accuracy"); axes[0,0].set_xlabel("Epochs"); axes[0,0].set_ylabel("Accuracy (%)"); axes[0,0].grid(True); axes[0,0].legend()
    axes[0,1].plot(naive_stats['test_loss'], label="Naive", marker='o'); axes[0,1].plot(iaq_stats['test_loss'], label="IAQ", marker='x'); axes[0,1].set_title("Test Loss"); axes[0,1].set_xlabel("Epochs"); axes[0,1].set_ylabel("Cross-Entropy Loss"); axes[0,1].grid(True); axes[0,1].legend()

    # Row 2: IAQ Dynamics
    ax = axes[1, 0]; ax2 = ax.twinx(); ax.plot(iaq_stats['avg_bits'], 'g-', label='Avg Total Bits'); ax2.plot(iaq_stats['rewards'], 'b-', label='Reward', alpha=0.6); ax.set_title("IAQ Training: Reward & Bit Usage"); ax.set_xlabel("Training Steps"); ax.set_ylabel("Average Bits", color='g'); ax2.set_ylabel("Reward", color='b'); ax.legend(loc='upper left'); ax2.legend(loc='upper right'); ax.grid(True)
    losses, bits = iaq_stats['final_test_bits_vs_loss']; sns.regplot(x=losses, y=bits, ax=axes[1,1], scatter_kws={'alpha':0.1, 's':10}, line_kws={'color':'red'}); axes[1,1].set_title("Bit Usage vs. Input 'Difficulty' (Loss)"); axes[1,1].set_xlabel("Per-Sample Loss"); axes[1,1].set_ylabel("Total Bits Used"); axes[1,1].grid(True)

    # Row 3: Bit Allocation Analysis
    bit_decisions = iaq_stats['final_bit_decisions']; df = pd.DataFrame(bit_decisions); df.columns = [f'L{i+1}' for i in df.columns]; sns.countplot(data=df.melt(var_name='Layer', value_name='Bits'), x='Layer', hue='Bits', ax=axes[2,0], palette='viridis'); axes[2,0].set_title("Final Bitwidth Choices per Layer"); axes[2,0].set_ylabel("Count")
    
    # NEW: Accuracy vs Bit Budget
    bits, correct = iaq_stats['final_bits_vs_correct']; df_corr = pd.DataFrame({'bits': bits, 'correct': correct}); sns.lineplot(data=df_corr, x='bits', y='correct', marker='o', ax=axes[2,1], errorbar='sd', color='purple'); axes[2,1].set_title("Accuracy vs. Bit Budget"); axes[2,1].set_xlabel("Total Bits Used per Sample"); axes[2,1].set_ylabel("Fraction Correct"); axes[2,1].grid(True)

    # Row 4: Controller Confidence and Heatmap
    sns.histplot(iaq_stats['final_controller_entropies'], ax=axes[3,0], kde=True, bins=30, color='orangered'); axes[3,0].set_title("Controller Confidence (Entropy)"); axes[3,0].set_xlabel("Entropy of Controller Output (Low = Confident)"); axes[3,0].grid(True)
    num_samples_to_show = 25; sample_decisions = pd.DataFrame(bit_decisions).iloc[:num_samples_to_show]; sns.heatmap(sample_decisions.T, ax=axes[3,1], cmap='viridis', cbar_kws={'label': 'Chosen Bitwidth'}); axes[3,1].set_title(f"Bit Allocation for First {num_samples_to_show} Samples"); axes[3,1].set_xlabel("Test Sample Index"); axes[3,1].set_ylabel("Layer Index")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(CONFIG['output_filename'], dpi=120)
    print(f"\n{C.GREEN}Visualizations saved to {C.BOLD}{CONFIG['output_filename']}{C.END}")
    plt.close() # Close the figure to free memory

# --- Main Execution ---
if __name__ == '__main__':
    print(f"{C.HEADER}{C.BOLD}Starting Experiment: Naive FFN vs. Input-Adaptive Quantization{C.END}")
    print(f"Using device: {C.BOLD}{CONFIG['device']}{C.END}")

    # --- Naive Model Run ---
    print(f"\n{C.BLUE}--- 1. Training Naive Baseline Model ---{C.END}")
    naive_stats_log = defaultdict(list)
    train_loader, test_loader = get_data_loaders(CONFIG['batch_size'])
    naive_model = NaiveFFN().to(CONFIG['device'])
    optimizer = torch.optim.Adam(naive_model.parameters(), lr=CONFIG['lr_main'])
    for epoch in range(1, CONFIG['epochs'] + 1):
        train_naive(naive_model, CONFIG['device'], train_loader, optimizer, epoch, naive_stats_log)
        test_naive(naive_model, CONFIG['device'], test_loader, naive_stats_log)
    
    # --- IAQ Model Run ---
    print(f"\n\n{C.BLUE}--- 2. Training Input-Adaptive Quantization (IAQ) Model ---{C.END}")
    iaq_stats_log = defaultdict(list)
    iaq_model = IAQ_FFN(bit_choices=CONFIG['bit_choices']).to(CONFIG['device'])
    main_params = [p for n, p in iaq_model.named_parameters() if 'controller' not in n]
    policy_params = [p for n, p in iaq_model.named_parameters() if 'controller' in n]
    optim_main = torch.optim.Adam(main_params, lr=CONFIG['lr_main'])
    optim_policy = torch.optim.Adam(policy_params, lr=CONFIG['lr_policy'])
    temp_schedule = np.linspace(CONFIG['gumbel_temp_initial'], CONFIG['gumbel_temp_final'], len(train_loader) * CONFIG['epochs'])
    for epoch in range(1, CONFIG['epochs'] + 1):
        train_iaq(iaq_model, CONFIG['device'], train_loader, optim_main, optim_policy, epoch, iaq_stats_log, temp_schedule)
        test_iaq(iaq_model, CONFIG['device'], test_loader, epoch, iaq_stats_log)

    # --- Final Analysis ---
    print(f"\n\n{C.HEADER}{C.BOLD}--- 3. Final Performance & Cost Analysis ---{C.END}")
    naive_perf_stats = calculate_model_stats(naive_model, 'naive')
    iaq_perf_stats = calculate_model_stats(iaq_model, 'iaq', iaq_stats_log)
    
    df = pd.DataFrame([naive_perf_stats, iaq_perf_stats], index=['Naive FFN', 'IAQ FFN'])
    df['Accuracy'] = [f"{naive_stats_log['test_acc'][-1]:.2f}%", f"{iaq_stats_log['test_acc'][-1]:.2f}%"]
    df_display = df[['Accuracy', 'params', 'size_mb', 'gflops', 'latency_ms']].copy()
    df_display.rename(columns={'params': 'Params', 'size_mb': 'Size (MB)', 'gflops': 'GFLOPs', 'latency_ms': 'Latency (ms/sample)'}, inplace=True)
    print(df_display.to_string(formatters={'Size (MB)': '{:.2f}'.format, 'GFLOPs': '{:.3f}'.format, 'Latency (ms/sample)': '{:.4f}'.format}))
    
    print(f"\n{C.CYAN}Breakdown of IAQ Parameters:{C.END}")
    print(f"  - Backbone: {iaq_perf_stats['params_backbone']:,}")
    print(f"  - Controllers: {iaq_perf_stats['params_controllers']:,} ({iaq_perf_stats['params_controllers']/iaq_perf_stats['params']*100:.2f}% of total)")

    # --- Generate and Save Report ---
    create_visualizations(naive_stats_log, iaq_stats_log)
