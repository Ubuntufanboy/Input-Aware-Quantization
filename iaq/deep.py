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
import time
import sys
import subprocess

# Check and install required packages
try:
    from tqdm import tqdm
except ImportError:
    print("tqdm not found. Installing..."); subprocess.check_call([sys.executable, "-m", "pip", "install", "tqdm"]); from tqdm import tqdm
try:
    import pandas as pd
except ImportError:
    print("Pandas not found. Installing..."); subprocess.check_call([sys.executable, "-m", "pip", "install", "pandas"]); import pandas as pd
try:
    import colorama
    colorama.init()
    class C:
        HEADER = '\033[95m'; BLUE = '\033[94m'; CYAN = '\033[96m'; GREEN = '\033[92m'
        YELLOW = '\033[93m'; RED = '\033[91m'; END = '\033[0m'; BOLD = '\033[1m'; UNDERLINE = '\033[4m'
except ImportError:
    class C: HEADER = BOLD = BLUE = CYAN = GREEN = YELLOW = RED = END = ''

# --- Configuration ---
CONFIG = {
    "epochs": 20, # Increased epochs for deeper network
    "batch_size": 256,
    "lr_main": 1e-3,
    "lr_policy": 5e-4, 
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "lambda_bits": 0.001, # Adjusted penalty for a deeper network
    "bit_choices": torch.tensor([2, 4, 8], dtype=torch.float32),
    "gumbel_temp_initial": 5.0,
    "gumbel_temp_final": 0.5,
    "output_filename": "deep_ffn_comparison_report2.png",
    "hidden_dim": 128,
    "num_hidden_layers": 14,
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

# --- NEW: Deep Naive FFN Model ---
class DeepFFN(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=128, output_dim=10, num_hidden_layers=14):
        super().__init__()
        layers = []
        # First layer
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.ReLU(inplace=True))
        
        # Hidden layers
        for _ in range(num_hidden_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU(inplace=True))
        
        self.hidden_layers = nn.Sequential(*layers)
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        
        # Kaiming Initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
                
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.hidden_layers(x)
        return self.output_layer(x)

# --- IAQ Model Components (QuantizedLayer is now a full Block) ---
class FakeQuantize(torch.autograd.Function): # Unchanged
    @staticmethod
    def forward(ctx, x, bits):
        if bits == 32: return x
        q_level = 2**bits - 1; min_val, max_val = x.min(), x.max(); scale = (max_val - min_val) / (q_level + 1e-9)
        dequantized = torch.round((x - min_val) / (scale + 1e-9)) * scale + min_val
        return dequantized
    @staticmethod
    def backward(ctx, grad_output): return grad_output, None
apply_fake_quant = FakeQuantize.apply

class QuantizationController(nn.Module): # Unchanged
    def __init__(self, num_choices):
        super().__init__(); self.controller_net = nn.Sequential(nn.Linear(3, 16), nn.ReLU(), nn.Linear(16, num_choices))
    def forward(self, x): return self.controller_net(x)

class QuantizedBlock(nn.Module):
    def __init__(self, in_features, out_features, bit_choices):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.bn = nn.BatchNorm1d(out_features)
        self.relu = nn.ReLU(inplace=True)
        self.bit_choices_values = bit_choices
        self.controller = QuantizationController(len(bit_choices))
        
        # Kaiming Init
        nn.init.kaiming_normal_(self.linear.weight, mode='fan_in', nonlinearity='relu')
        if self.linear.bias is not None: nn.init.constant_(self.linear.bias, 0)

    def forward(self, x, temp, is_eval=False):
        batch_size = x.size(0)
        with torch.no_grad():
            stats = torch.stack([x.mean(dim=1), x.var(dim=1), (x == 0).float().mean(dim=1)], dim=1)
        logits = self.controller(stats)
        if is_eval:
            choice_indices = logits.argmax(dim=1); log_probs = torch.zeros(batch_size, device=x.device)
        else:
            gumbel_probs = F.gumbel_softmax(logits, tau=temp, hard=True)
            choice_indices = gumbel_probs.argmax(dim=1)
            log_probs = torch.log(F.softmax(logits, dim=-1).gather(1, choice_indices.unsqueeze(1)).squeeze() + 1e-9)
        chosen_bits = self.bit_choices_values[choice_indices]
        
        output = torch.zeros(batch_size, self.linear.out_features, device=x.device)
        for bit_val in chosen_bits.unique():
            mask = (chosen_bits == bit_val)
            quantized_weight = apply_fake_quant(self.linear.weight, bit_val)
            output[mask] = F.linear(x[mask], quantized_weight, self.linear.bias)
        
        output = self.relu(self.bn(output))
        return output, log_probs, chosen_bits

class IAQ_DeepFFN(nn.Module):
    def __init__(self, bit_choices, input_dim=784, hidden_dim=128, output_dim=10, num_hidden_layers=14):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(QuantizedBlock(input_dim, hidden_dim, bit_choices))
        for _ in range(num_hidden_layers - 1):
            self.layers.append(QuantizedBlock(hidden_dim, hidden_dim, bit_choices))
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        nn.init.kaiming_normal_(self.output_layer.weight, mode='fan_in', nonlinearity='relu')
        if self.output_layer.bias is not None: nn.init.constant_(self.output_layer.bias, 0)

    def forward(self, x, temp=1.0, is_eval=False):
        x = x.view(x.size(0), -1)
        batch_log_probs, batch_bits = [], []
        for layer in self.layers:
            x, log_p, bits = layer(x, temp, is_eval)
            batch_log_probs.append(log_p); batch_bits.append(bits)
        log_probs_per_sample = torch.stack(batch_log_probs, dim=1).sum(dim=1)
        bits_per_sample = torch.stack(batch_bits, dim=1)
        output = self.output_layer(x)
        return output, log_probs_per_sample, bits_per_sample

# --- Training, Test, and Stat Functions (largely unchanged, but will work with new models) ---
def train_naive(model, device, train_loader, optimizer, epoch, stats):
    model.train()
    pbar = tqdm(train_loader, desc=f"{C.CYAN}Epoch {epoch} [Train]{C.END}", unit="batch")
    for data, target in pbar:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad(); output = model(data); loss = F.cross_entropy(output, target)
        loss.backward(); optimizer.step(); pbar.set_postfix(loss=f"{loss.item():.4f}")
    stats['train_loss'].append(loss.item())

def test_naive(model, device, test_loader, stats, epoch): # Added epoch for consistent printing
    model.eval(); test_loss = 0; correct = 0
    pbar = tqdm(test_loader, desc=f"{C.YELLOW}Epoch {epoch} [Test]{C.END}", unit="batch")
    with torch.no_grad():
        for data, target in pbar:
            data, target = data.to(device), target.to(device)
            output = model(data); test_loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True); correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset); accuracy = 100. * correct / len(test_loader.dataset)
    stats['test_loss'].append(test_loss); stats['test_acc'].append(accuracy)
    print(f"  {C.YELLOW}└> Test Results: Avg loss: {test_loss:.4f}, Accuracy: {accuracy:.2f}%{C.END}")

def train_iaq(model, device, train_loader, optim_main, optim_policy, epoch, stats, temp_schedule):
    model.train()
    pbar = tqdm(train_loader, desc=f"{C.CYAN}Epoch {epoch} [Train]{C.END}", unit="batch")
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        global_step = (epoch - 1) * len(train_loader) + batch_idx; temp = temp_schedule[global_step]
        optim_main.zero_grad(); optim_policy.zero_grad()
        output, log_probs, bits_per_sample = model(data, temp=temp)
        task_loss = F.cross_entropy(output, target)
        total_bits_per_sample = bits_per_sample.sum(dim=1)
        bit_cost = CONFIG['lambda_bits'] * total_bits_per_sample
        per_sample_task_loss = F.cross_entropy(output, target, reduction='none')
        reward = -per_sample_task_loss.detach() - bit_cost
        policy_loss = (-log_probs * reward).mean()
        total_loss = task_loss + policy_loss
        total_loss.backward(); optim_main.step(); optim_policy.step()
        stats['rewards'].append(reward.mean().item()); stats['avg_bits'].append(total_bits_per_sample.mean().item()); stats['train_loss'].append(task_loss.item())
        pbar.set_postfix(task_loss=f"{task_loss.item():.3f}", reward=f"{reward.mean().item():.3f}", avg_bits=f"{total_bits_per_sample.mean().item():.1f}")

def test_iaq(model, device, test_loader, epoch, stats):
    model.eval(); test_loss = 0; correct = 0
    pbar = tqdm(test_loader, desc=f"{C.YELLOW}Epoch {epoch} [Test]{C.END}", unit="batch")
    all_bits, all_losses, all_bit_decisions, all_correct, all_entropies = [], [], defaultdict(list), [], []
    with torch.no_grad():
        for data, target in pbar:
            data, target = data.to(device), target.to(device)
            output, _, bits_per_sample = model(data, is_eval=True)
            per_sample_loss = F.cross_entropy(output, target, reduction='none'); test_loss += per_sample_loss.sum().item()
            pred = output.argmax(dim=1); is_correct = pred.eq(target); correct += is_correct.sum().item()
            total_bits_per_sample = bits_per_sample.sum(dim=1)
            all_bits.extend(total_bits_per_sample.cpu().tolist()); all_losses.extend(per_sample_loss.cpu().tolist()); all_correct.extend(is_correct.cpu().tolist())
            for i in range(bits_per_sample.shape[1]): all_bit_decisions[i].extend(bits_per_sample[:, i].cpu().tolist())
            for layer in model.layers:
                logits = layer.controller(torch.stack([data.view(data.size(0),-1).mean(dim=1), data.view(data.size(0),-1).var(dim=1), (data==0).float().mean(dim=(1,2,3))], dim=1))
                entropy = -(F.softmax(logits, dim=1) * F.log_softmax(logits, dim=1)).sum(dim=1)
                all_entropies.extend(entropy.cpu().tolist())
    test_loss /= len(test_loader.dataset); accuracy = 100. * correct / len(test_loader.dataset); avg_total_bits = np.mean(all_bits)
    stats['test_loss'].append(test_loss); stats['test_acc'].append(accuracy); stats['final_test_bits_vs_loss'] = (all_losses, all_bits); stats['final_bit_decisions'] = all_bit_decisions; stats['final_bits_vs_correct'] = (all_bits, all_correct); stats['final_controller_entropies'] = all_entropies
    print(f"  {C.YELLOW}└> Test Results: Avg loss: {test_loss:.4f}, Accuracy: {accuracy:.2f}%, Avg Bits: {avg_total_bits:.2f}{C.END}")

def calculate_model_stats(model, model_type, iaq_stats=None):
    stats = {}
    params_backbone = sum(p.numel() for n, p in model.named_parameters() if 'controller' not in n)
    if model_type == 'naive':
        stats['params'] = params_backbone; stats['params_controllers'] = 0
        stats['size_mb'] = stats['params'] * 4 / (1024**2)
    else: # IAQ
        params_controllers = sum(p.numel() for n, p in model.named_parameters() if 'controller' in n)
        stats['params'] = params_backbone + params_controllers; stats['params_controllers'] = params_controllers
        avg_bits_per_weight = np.mean(list(iaq_stats['final_bit_decisions'].values()))
        backbone_size_bits = params_backbone * avg_bits_per_weight; controller_size_bits = params_controllers * 32
        stats['size_mb'] = (backbone_size_bits + controller_size_bits) / (8 * 1024**2)
    
    total_flops = sum(2 * m.in_features * m.out_features for m in model.modules() if isinstance(m, nn.Linear))
    stats['gflops'] = total_flops / 1e9
    return stats

def create_visualizations(naive_stats, iaq_stats):
    fig, axes = plt.subplots(4, 2, figsize=(18, 28))
    fig.suptitle('Deep-Narrow FFN (15 Layers) vs. IAQ Deep-Narrow FFN', fontsize=22, weight='bold')
    # All plotting code is identical to previous version, it will adapt to the new data
    axes[0,0].plot(naive_stats['test_acc'], label=f"Naive (Final: {naive_stats['test_acc'][-1]:.2f}%)", marker='o'); axes[0,0].plot(iaq_stats['test_acc'], label=f"IAQ (Final: {iaq_stats['test_acc'][-1]:.2f}%)", marker='x'); axes[0,0].set_title("Test Accuracy"); axes[0,0].set_xlabel("Epochs"); axes[0,0].set_ylabel("Accuracy (%)"); axes[0,0].grid(True); axes[0,0].legend()
    axes[0,1].plot(naive_stats['test_loss'], label="Naive", marker='o'); axes[0,1].plot(iaq_stats['test_loss'], label="IAQ", marker='x'); axes[0,1].set_title("Test Loss"); axes[0,1].set_xlabel("Epochs"); axes[0,1].set_ylabel("Cross-Entropy Loss"); axes[0,1].grid(True); axes[0,1].legend()
    ax = axes[1, 0]; ax2 = ax.twinx(); ax.plot(iaq_stats['avg_bits'], 'g-', label='Avg Total Bits'); ax2.plot(iaq_stats['rewards'], 'b-', label='Reward', alpha=0.6); ax.set_title("IAQ Training Dynamics"); ax.set_xlabel("Training Steps"); ax.set_ylabel("Average Bits", color='g'); ax2.set_ylabel("Reward", color='b'); ax.legend(loc='upper left'); ax2.legend(loc='upper right'); ax.grid(True)
    losses, bits = iaq_stats['final_test_bits_vs_loss']; sns.regplot(x=losses, y=bits, ax=axes[1,1], scatter_kws={'alpha':0.1, 's':10}, line_kws={'color':'red'}); axes[1,1].set_title("Bit Usage vs. Input 'Difficulty' (Loss)"); axes[1,1].set_xlabel("Per-Sample Loss"); axes[1,1].set_ylabel("Total Bits Used"); axes[1,1].grid(True)
    bit_decisions = iaq_stats['final_bit_decisions']; df = pd.DataFrame(bit_decisions); df.columns = [f'L{i+1}' for i in df.columns]; sns.countplot(data=df.melt(var_name='Layer', value_name='Bits'), x='Layer', hue='Bits', ax=axes[2,0], palette='viridis'); axes[2,0].set_title("Final Bitwidth Choices per Layer"); axes[2,0].set_ylabel("Count"); axes[2,0].tick_params(axis='x', rotation=45)
    bits, correct = iaq_stats['final_bits_vs_correct']; df_corr = pd.DataFrame({'bits': bits, 'correct': correct}); sns.lineplot(data=df_corr, x='bits', y='correct', marker='o', ax=axes[2,1], errorbar='sd', color='purple'); axes[2,1].set_title("Accuracy vs. Bit Budget"); axes[2,1].set_xlabel("Total Bits Used per Sample"); axes[2,1].set_ylabel("Fraction Correct"); axes[2,1].grid(True)
    sns.histplot(iaq_stats['final_controller_entropies'], ax=axes[3,0], kde=True, bins=30, color='orangered'); axes[3,0].set_title("Controller Confidence (Entropy)"); axes[3,0].set_xlabel("Entropy of Controller Output (Low = Confident)"); axes[3,0].grid(True)
    num_samples_to_show = 15; sample_decisions = pd.DataFrame(bit_decisions).T.iloc[:,:num_samples_to_show]; sns.heatmap(sample_decisions, ax=axes[3,1], cmap='viridis', cbar_kws={'label': 'Chosen Bitwidth'}); axes[3,1].set_title(f"Bit Allocation for First {num_samples_to_show} Samples"); axes[3,1].set_xlabel("Test Sample Index"); axes[3,1].set_ylabel("Layer Index")
    plt.tight_layout(rect=[0, 0, 1, 0.97]); plt.savefig(CONFIG['output_filename'], dpi=120)
    print(f"\n{C.GREEN}Visualizations saved to {C.BOLD}{CONFIG['output_filename']}{C.END}"); plt.close()

# --- Main Execution ---
if __name__ == '__main__':
    print(f"{C.HEADER}{C.BOLD}Starting Experiment: DEEP-NARROW FFN (15 Layers){C.END}")
    print(f"Using device: {C.BOLD}{CONFIG['device']}{C.END}")
    # --- Naive Model Run ---
    print(f"\n{C.BLUE}--- 1. Training Naive Deep FFN Model ---{C.END}")
    naive_stats_log = defaultdict(list)
    train_loader, test_loader = get_data_loaders(CONFIG['batch_size'])
    naive_model = DeepFFN(hidden_dim=CONFIG['hidden_dim'], num_hidden_layers=CONFIG['num_hidden_layers']).to(CONFIG['device'])
    optimizer = torch.optim.Adam(naive_model.parameters(), lr=CONFIG['lr_main'])
    for epoch in range(1, CONFIG['epochs'] + 1):
        train_naive(naive_model, CONFIG['device'], train_loader, optimizer, epoch, naive_stats_log)
        test_naive(naive_model, CONFIG['device'], test_loader, naive_stats_log, epoch)
    # --- IAQ Model Run ---
    print(f"\n\n{C.BLUE}--- 2. Training IAQ Deep FFN Model ---{C.END}")
    iaq_stats_log = defaultdict(list)
    iaq_model = IAQ_DeepFFN(bit_choices=CONFIG['bit_choices'], hidden_dim=CONFIG['hidden_dim'], num_hidden_layers=CONFIG['num_hidden_layers']).to(CONFIG['device'])
    main_params = [p for n, p in iaq_model.named_parameters() if 'controller' not in n]; policy_params = [p for n, p in iaq_model.named_parameters() if 'controller' in n]
    optim_main = torch.optim.Adam(main_params, lr=CONFIG['lr_main']); optim_policy = torch.optim.Adam(policy_params, lr=CONFIG['lr_policy'])
    temp_schedule = np.linspace(CONFIG['gumbel_temp_initial'], CONFIG['gumbel_temp_final'], len(train_loader) * CONFIG['epochs'])
    for epoch in range(1, CONFIG['epochs'] + 1):
        train_iaq(iaq_model, CONFIG['device'], train_loader, optim_main, optim_policy, epoch, iaq_stats_log, temp_schedule)
        test_iaq(iaq_model, CONFIG['device'], test_loader, epoch, iaq_stats_log)
    # --- Final Analysis ---
    print(f"\n\n{C.HEADER}{C.BOLD}--- 3. Final Performance & Cost Analysis ---{C.END}")
    naive_perf_stats = calculate_model_stats(naive_model, 'naive'); iaq_perf_stats = calculate_model_stats(iaq_model, 'iaq', iaq_stats_log)
    df = pd.DataFrame([naive_perf_stats, iaq_perf_stats], index=['Deep Naive FFN', 'Deep IAQ FFN'])
    df['Accuracy'] = [f"{naive_stats_log['test_acc'][-1]:.2f}%", f"{iaq_stats_log['test_acc'][-1]:.2f}%"]
    df_display = df[['Accuracy', 'params', 'size_mb', 'gflops']].copy()
    df_display.rename(columns={'params': 'Params', 'size_mb': 'Size (MB)', 'gflops': 'GFLOPs'}, inplace=True)
    print(df_display.to_string(formatters={'Size (MB)': '{:.2f}'.format, 'GFLOPs': '{:.3f}'.format}))
    print(f"\n{C.CYAN}Breakdown of IAQ Parameters:{C.END}")
    print(f"  - Backbone: {iaq_perf_stats['params']-iaq_perf_stats['params_controllers']:,}")
    print(f"  - Controllers: {iaq_perf_stats['params_controllers']:,} ({iaq_perf_stats['params_controllers']/iaq_perf_stats['params']*100:.2f}% of total)")
    create_visualizations(naive_stats_log, iaq_stats_log)
