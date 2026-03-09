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

# Package checks and color class...
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
    # THIS IS THE CORRECTED LINE
    class C:
        HEADER = '\033[95m'; BLUE = '\033[94m'; CYAN = '\033[96m'; GREEN = '\033[92m'
        YELLOW = '\033[93m'; RED = '\033[91m'; END = '\033[0m'; BOLD = '\033[1m'; UNDERLINE = '\033[4m'
except ImportError:
    class C: HEADER = BOLD = BLUE = CYAN = GREEN = YELLOW = RED = END = ''


# --- Configuration ---
CONFIG = {
    "epochs": 50,
    "batch_size": 256,
    "lr": 1e-3, # Unified learning rate
    "lr_policy": 5e-4, 
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "lambda_bits": 0.0005, 
    "bit_choices": torch.tensor([2, 4, 8], dtype=torch.float32),
    "gumbel_temp_initial": 5.0,
    "gumbel_temp_final": 0.5,
    "output_filename": "3_way_deep_ffn_report.png",
    "hidden_dim": 128,
    "num_hidden_layers": 14,
}
CONFIG['bit_choices'] = CONFIG['bit_choices'].to(CONFIG['device'])

# --- Utilities ---
def get_data_loaders(batch_size, shuffle_train=True):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    return DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle_train, num_workers=2, pin_memory=True), \
           DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

class FakeQuantize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, bits):
        if bits == 32: return x
        q_level = 2**bits - 1
        min_val, max_val = x.min(), x.max()
        scale = (max_val - min_val) / (q_level + 1e-9)
        dequantized = torch.round((x - min_val) / (scale + 1e-9)) * scale + min_val
        return dequantized
    @staticmethod
    def backward(ctx, grad_output): return grad_output, None
apply_fake_quant = FakeQuantize.apply

def _init_weights(module):
    if isinstance(module, nn.Linear):
        nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
        if module.bias is not None: nn.init.constant_(module.bias, 0)

# --- MODEL DEFINITIONS ---

# 1. Deep Naive FFN (FP32)
class DeepFFN(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=128, output_dim=10, num_hidden_layers=14):
        super().__init__()
        layers = [nn.Linear(input_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(inplace=True)]
        for _ in range(num_hidden_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(inplace=True)])
        self.hidden_layers = nn.Sequential(*layers)
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        self.apply(_init_weights)
    def forward(self, x):
        return self.output_layer(self.hidden_layers(x.view(x.size(0), -1)))

# 2. Deep QAT FFN (Static 4-bit)
class QAT_DeepFFN(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=128, output_dim=10, num_hidden_layers=14, static_bits=4):
        super().__init__()
        self.static_bits = static_bits
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_dim, hidden_dim))
        for _ in range(num_hidden_layers - 1):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.bns = nn.ModuleList([nn.BatchNorm1d(hidden_dim) for _ in range(num_hidden_layers)])
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        self.apply(_init_weights)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        for i, layer in enumerate(self.layers):
            quantized_weight = apply_fake_quant(layer.weight, self.static_bits)
            x = F.linear(x, quantized_weight, layer.bias)
            x = F.relu(self.bns[i](x))
        return self.output_layer(x)

# 3. Deep IAQ FFN (Dynamic 2/4/8-bit)
class QuantizationController(nn.Module):
    def __init__(self, num_choices):
        super().__init__(); self.controller_net = nn.Sequential(nn.Linear(3, 16), nn.ReLU(), nn.Linear(16, num_choices))
    def forward(self, x): return self.controller_net(x)

class QuantizedBlock(nn.Module):
    def __init__(self, in_features, out_features, bit_choices):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features); self.bn = nn.BatchNorm1d(out_features)
        self.relu = nn.ReLU(inplace=True); self.bit_choices_values = bit_choices
        self.controller = QuantizationController(len(bit_choices)); self.apply(_init_weights)
    def forward(self, x, temp, is_eval=False):
        batch_size = x.size(0)
        with torch.no_grad(): stats = torch.stack([x.mean(dim=1), x.var(dim=1), (x == 0).float().mean(dim=1)], dim=1)
        logits = self.controller(stats)
        if is_eval:
            choice_indices = logits.argmax(dim=1); log_probs = torch.zeros(batch_size, device=x.device)
        else:
            gumbel_probs = F.gumbel_softmax(logits, tau=temp, hard=True); choice_indices = gumbel_probs.argmax(dim=1)
            log_probs = torch.log(F.softmax(logits, dim=-1).gather(1, choice_indices.unsqueeze(1)).squeeze() + 1e-9)
        chosen_bits = self.bit_choices_values[choice_indices]
        output = torch.zeros(batch_size, self.linear.out_features, device=x.device)
        for bit_val in chosen_bits.unique():
            mask = (chosen_bits == bit_val)
            quantized_weight = apply_fake_quant(self.linear.weight, bit_val)
            output[mask] = F.linear(x[mask], quantized_weight, self.linear.bias)
        return self.relu(self.bn(output)), log_probs, chosen_bits

class IAQ_DeepFFN(nn.Module):
    def __init__(self, bit_choices, input_dim=784, hidden_dim=128, output_dim=10, num_hidden_layers=14):
        super().__init__()
        self.layers = nn.ModuleList([QuantizedBlock(input_dim if i==0 else hidden_dim, hidden_dim, bit_choices) for i in range(num_hidden_layers)])
        self.output_layer = nn.Linear(hidden_dim, output_dim); self.apply(_init_weights)
    def forward(self, x, temp=1.0, is_eval=False):
        x = x.view(x.size(0), -1); batch_log_probs, batch_bits = [], []
        for layer in self.layers:
            x, log_p, bits = layer(x, temp, is_eval); batch_log_probs.append(log_p); batch_bits.append(bits)
        return self.output_layer(x), torch.stack(batch_log_probs, dim=1).sum(dim=1), torch.stack(batch_bits, dim=1)

# --- Generic Training and Testing Loops ---
def train(model, loader, optimizer, epoch, model_type, stats_log, **kwargs):
    model.train(); pbar = tqdm(loader, desc=f"{C.CYAN}Epoch {epoch} [Train {model_type}]{C.END}", unit="batch")
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(CONFIG['device']), target.to(CONFIG['device'])
        optimizer.zero_grad()
        if model_type == 'IAQ':
            optim_policy = kwargs['optim_policy']
            optim_policy.zero_grad()
            temp_schedule = kwargs['temp_schedule']
            global_step = (epoch - 1) * len(loader) + batch_idx; temp = temp_schedule[global_step]
            output, log_probs, bits_per_sample = model(data, temp=temp)
            task_loss = F.cross_entropy(output, target)
            total_bits = bits_per_sample.sum(dim=1); bit_cost = CONFIG['lambda_bits'] * total_bits
            reward = -F.cross_entropy(output, target, reduction='none').detach() - bit_cost
            policy_loss = (-log_probs * reward).mean(); total_loss = task_loss + policy_loss
            total_loss.backward(); optimizer.step(); optim_policy.step()
            stats_log['rewards'].append(reward.mean().item()); stats_log['avg_bits'].append(total_bits.mean().item())
            pbar.set_postfix(loss=f"{task_loss.item():.3f}", reward=f"{reward.mean().item():.3f}", bits=f"{total_bits.mean().item():.1f}")
        else:
            output = model(data); loss = F.cross_entropy(output, target)
            loss.backward(); optimizer.step()
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        stats_log['train_loss'].append(loss.item() if model_type != 'IAQ' else task_loss.item())

def test(model, loader, epoch, model_type, stats_log):
    model.eval(); test_loss = 0; correct = 0
    pbar = tqdm(loader, desc=f"{C.YELLOW}Epoch {epoch} [Test {model_type}]{C.END}", unit="batch")
    all_bits, all_losses = [], []
    with torch.no_grad():
        for data, target in pbar:
            data, target = data.to(CONFIG['device']), target.to(CONFIG['device'])
            if model_type == 'IAQ':
                output, _, bits_per_sample = model(data, is_eval=True)
                total_bits_per_sample = bits_per_sample.sum(dim=1)
                all_bits.extend(total_bits_per_sample.cpu().tolist())
                all_losses.extend(F.cross_entropy(output, target, reduction='none').cpu().tolist())
                if 'final_bit_decisions' not in stats_log: stats_log['final_bit_decisions'] = defaultdict(list)
                for i in range(bits_per_sample.shape[1]): stats_log['final_bit_decisions'][i].extend(bits_per_sample[:, i].cpu().tolist())
            else:
                output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()
            correct += output.argmax(dim=1).eq(target).sum().item()
    test_loss /= len(loader.dataset); accuracy = 100. * correct / len(loader.dataset)
    stats_log['test_loss'].append(test_loss); stats_log['test_acc'].append(accuracy)
    avg_bits_str = f", Avg Bits: {np.mean(all_bits):.2f}" if model_type == 'IAQ' else ""
    print(f"  {C.YELLOW}└> Test Results: Avg loss: {test_loss:.4f}, Accuracy: {accuracy:.2f}%{avg_bits_str}{C.END}")
    if model_type == 'IAQ': stats_log['final_test_bits_vs_loss'] = (all_losses, all_bits)

# --- Analysis & Visualization ---
def calculate_model_stats(model, model_type, iaq_stats=None):
    stats = {}
    params_backbone = sum(p.numel() for n, p in model.named_parameters() if 'controller' not in n)
    if model_type == 'Naive':
        stats['params'] = params_backbone; stats['size_mb'] = stats['params'] * 4 / (1024**2)
    elif model_type == 'QAT':
        stats['params'] = params_backbone; stats['size_mb'] = stats['params'] * model.static_bits / (8 * 1024**2)
    else: # IAQ
        params_controllers = sum(p.numel() for n, p in model.named_parameters() if 'controller' in n)
        stats['params'] = params_backbone + params_controllers
        avg_bits_per_weight = np.mean(list(iaq_stats['final_bit_decisions'].values()))
        stats['size_mb'] = (params_backbone * avg_bits_per_weight + params_controllers * 32) / (8 * 1024**2)
    stats['gflops'] = sum(2 * m.in_features * m.out_features for m in model.modules() if isinstance(m, nn.Linear)) / 1e9
    return stats

def create_visualizations(all_stats):
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    fig.suptitle('Deep FFN (15 Layers): FP32 vs. Static 4-bit QAT vs. Dynamic IAQ', fontsize=22, weight='bold')
    colors = {'Naive': 'blue', 'QAT': 'orange', 'IAQ': 'green'}
    markers = {'Naive': 'o', 'QAT': 's', 'IAQ': 'x'}
    
    for name, stats in all_stats.items():
        axes[0,0].plot(stats['test_acc'], label=f"{name} (Final: {stats['test_acc'][-1]:.2f}%)", color=colors[name], marker=markers[name])
        axes[0,1].plot(stats['test_loss'], label=name, color=colors[name], marker=markers[name])
    axes[0,0].set_title("Test Accuracy Comparison"); axes[0,0].set_xlabel("Epochs"); axes[0,0].set_ylabel("Accuracy (%)"); axes[0,0].grid(True); axes[0,0].legend()
    axes[0,1].set_title("Test Loss Comparison"); axes[0,1].set_xlabel("Epochs"); axes[0,1].set_ylabel("Cross-Entropy Loss"); axes[0,1].grid(True); axes[0,1].legend()

    iaq_stats = all_stats['IAQ']
    bit_decisions = iaq_stats['final_bit_decisions']
    df = pd.DataFrame(bit_decisions); df.columns = [f'L{i+1}' for i in df.columns]
    sns.countplot(data=df.melt(var_name='Layer', value_name='Bits'), x='Layer', hue='Bits', ax=axes[1,0], palette='viridis')
    axes[1,0].set_title("IAQ: Final Bitwidth Choices per Layer"); axes[1,0].set_ylabel("Count"); axes[1,0].tick_params(axis='x', rotation=45)
    
    losses, bits = iaq_stats['final_test_bits_vs_loss']
    sns.regplot(x=losses, y=bits, ax=axes[1,1], scatter_kws={'alpha':0.1, 's':10}, line_kws={'color':'red'})
    axes[1,1].set_title("IAQ: Bit Usage vs. Input 'Difficulty' (Loss)"); axes[1,1].set_xlabel("Per-Sample Loss"); axes[1,1].set_ylabel("Total Bits Used"); axes[1,1].grid(True)

    plt.tight_layout(rect=[0, 0, 1, 0.96]); plt.savefig(CONFIG['output_filename'], dpi=120)
    print(f"\n{C.GREEN}Visualizations saved to {C.BOLD}{CONFIG['output_filename']}{C.END}"); plt.close()

# --- Main Execution ---
if __name__ == '__main__':
    print(f"{C.HEADER}{C.BOLD}Starting 3-Way Experiment: DEEP-NARROW FFN (15 Layers){C.END}")
    print(f"Using device: {C.BOLD}{CONFIG['device']}{C.END}\n")
    
    all_stats_logs = defaultdict(lambda: defaultdict(list))
    models = {}
    
    # --- Model Initialization ---
    models['Naive'] = DeepFFN(hidden_dim=CONFIG['hidden_dim'], num_hidden_layers=CONFIG['num_hidden_layers']).to(CONFIG['device'])
    models['QAT'] = QAT_DeepFFN(hidden_dim=CONFIG['hidden_dim'], num_hidden_layers=CONFIG['num_hidden_layers'], static_bits=4).to(CONFIG['device'])
    models['IAQ'] = IAQ_DeepFFN(bit_choices=CONFIG['bit_choices'], hidden_dim=CONFIG['hidden_dim'], num_hidden_layers=CONFIG['num_hidden_layers']).to(CONFIG['device'])
    
    # --- Training Loop ---
    for name, model in models.items():
        print(f"{C.BLUE}--- Training {name} Model ---{C.END}")
        train_loader, test_loader = get_data_loaders(CONFIG['batch_size'])
        if name == 'IAQ':
            main_params = [p for n, p in model.named_parameters() if 'controller' not in n]
            policy_params = [p for n, p in model.named_parameters() if 'controller' in n]
            optimizer = torch.optim.Adam(main_params, lr=CONFIG['lr'])
            optim_policy = torch.optim.Adam(policy_params, lr=CONFIG['lr_policy'])
            temp_schedule = np.linspace(CONFIG['gumbel_temp_initial'], CONFIG['gumbel_temp_final'], len(train_loader) * CONFIG['epochs'])
            train_kwargs = {'optim_policy': optim_policy, 'temp_schedule': temp_schedule}
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['lr'])
            train_kwargs = {}
        
        for epoch in range(1, CONFIG['epochs'] + 1):
            train(model, train_loader, optimizer, epoch, name, all_stats_logs[name], **train_kwargs)
            test(model, test_loader, epoch, name, all_stats_logs[name])
        print("\n")

    # --- Final Analysis ---
    print(f"\n{C.HEADER}{C.BOLD}--- Final Performance & Cost Analysis ---{C.END}")
    perf_stats_list = []
    for name, model in models.items():
        stats = calculate_model_stats(model, name, all_stats_logs[name])
        stats['Accuracy'] = f"{all_stats_logs[name]['test_acc'][-1]:.2f}%"
        stats['Model'] = f"{name} FFN"
        perf_stats_list.append(stats)

    df = pd.DataFrame(perf_stats_list).set_index('Model')
    df_display = df[['Accuracy', 'params', 'size_mb', 'gflops']].copy()
    df_display.rename(columns={'params': 'Params', 'size_mb': 'Effective Size (MB)', 'gflops': 'GFLOPs'}, inplace=True)
    print(df_display.to_string(formatters={'Effective Size (MB)': '{:.2f}'.format, 'GFLOPs': '{:.3f}'.format}))
    
    create_visualizations(all_stats_logs)
