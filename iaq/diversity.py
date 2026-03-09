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
    class C:
        HEADER = '\033[95m'; BLUE = '\033[94m'; CYAN = '\033[96m'; GREEN = '\033[92m'
        YELLOW = '\033[93m'; RED = '\033[91m'; END = '\033[0m'; BOLD = '\033[1m'; UNDERLINE = '\033[4m'
except ImportError:
    class C: HEADER = BOLD = BLUE = CYAN = GREEN = YELLOW = RED = END = ''


# --- Configuration ---
CONFIG = {
    "epochs": 50,
    "batch_size": 256,
    "lr": 1e-3,
    "lr_policy": 5e-4,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "bit_choices": torch.tensor([2, 4, 8], dtype=torch.float32),
    "gumbel_temp_initial": 5.0,
    "gumbel_temp_final": 0.5,
    "output_filename": "diversity_iaq_cifar10_report.png",
    "hidden_dim": 128,
    "num_hidden_layers": 14,
    "lambda_bits": 0.0005,
    "gamma_sensitivity": 0.05,
    "input_dim": 32 * 32 * 3,

    # --- NEW: Hyperparameter for Diversity-Driven Exploration ---
    "eta_diversity": 0.1, # Controls the strength of the diversity bonus. This is a key knob to tune!
}
CONFIG['bit_choices'] = CONFIG['bit_choices'].to(CONFIG['device'])

# --- Utilities (CIFAR-10 data loaders) ---
def get_data_loaders(batch_size, shuffle_train=True):
    cifar10_mean = [0.4914, 0.4822, 0.4465]
    cifar10_std = [0.2023, 0.1994, 0.2010]
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), transforms.Normalize(cifar10_mean, cifar10_std),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(), transforms.Normalize(cifar10_mean, cifar10_std),
    ])
    train_dataset = datasets.CIFAR10('./data', train=True, download=True, transform=transform_train)
    test_dataset = datasets.CIFAR10('./data', train=False, transform=transform_test)
    return DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle_train, num_workers=4, pin_memory=True), \
           DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

class FakeQuantize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, bits):
        if bits == 32: return x
        q_level = 2**bits - 1; max_val = torch.max(torch.abs(x))
        scale = (2 * max_val) / (q_level + 1e-9)
        return torch.round(x / (scale + 1e-9)) * scale
    @staticmethod
    def backward(ctx, grad_output): return grad_output, None
apply_fake_quant = FakeQuantize.apply

def _init_weights(module):
    if isinstance(module, nn.Linear):
        nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
        if module.bias is not None: nn.init.constant_(module.bias, 0)

# --- MODEL DEFINITIONS (Unchanged) ---
class DeepFFN(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, output_dim=10, num_hidden_layers=14):
        super().__init__(); layers = [nn.Linear(input_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(inplace=True)]
        for _ in range(num_hidden_layers - 1): layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(inplace=True)])
        self.hidden_layers = nn.Sequential(*layers); self.output_layer = nn.Linear(hidden_dim, output_dim); self.apply(_init_weights)
    def forward(self, x): return self.output_layer(self.hidden_layers(x.view(x.size(0), -1)))

class QAT_DeepFFN(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, output_dim=10, num_hidden_layers=14, static_bits=4):
        super().__init__(); self.static_bits = static_bits; self.layers = nn.ModuleList([nn.Linear(input_dim if i==0 else hidden_dim, hidden_dim) for i in range(num_hidden_layers)])
        self.bns = nn.ModuleList([nn.BatchNorm1d(hidden_dim) for _ in range(num_hidden_layers)]); self.output_layer = nn.Linear(hidden_dim, output_dim); self.apply(_init_weights)
    def forward(self, x):
        x = x.view(x.size(0), -1)
        for i, layer in enumerate(self.layers):
            quantized_weight = apply_fake_quant(layer.weight, torch.tensor(self.static_bits)); x = F.relu(self.bns[i](F.linear(x, quantized_weight, layer.bias)))
        return self.output_layer(x)

class QuantizationController(nn.Module):
    def __init__(self, num_choices):
        super().__init__(); self.controller_net = nn.Sequential(nn.Linear(3, 16), nn.ReLU(), nn.Linear(16, num_choices))
    def forward(self, x): return self.controller_net(x)

class QuantizedBlock(nn.Module):
    def __init__(self, in_features, out_features, bit_choices):
        super().__init__(); self.linear = nn.Linear(in_features, out_features); self.bn = nn.BatchNorm1d(out_features)
        self.relu = nn.ReLU(inplace=True); self.bit_choices_values = bit_choices; self.controller = QuantizationController(len(bit_choices)); self.apply(_init_weights)
    def forward(self, x, temp, is_eval=False):
        batch_size = x.size(0);
        with torch.no_grad(): stats = torch.stack([x.mean(dim=1), x.var(dim=1), (x == 0).float().mean(dim=1)], dim=1)
        logits = self.controller(stats)
        if is_eval: choice_indices, log_probs = logits.argmax(dim=1), None
        else:
            gumbel_probs = F.gumbel_softmax(logits, tau=temp, hard=True); choice_indices = gumbel_probs.argmax(dim=1)
            log_probs = torch.log(F.softmax(logits, dim=-1).gather(1, choice_indices.unsqueeze(1)).squeeze() + 1e-9)
        chosen_bits = self.bit_choices_values[choice_indices]; output = torch.zeros(batch_size, self.linear.out_features, device=x.device)
        for bit_val in chosen_bits.unique():
            mask = (chosen_bits == bit_val); quantized_weight = apply_fake_quant(self.linear.weight, bit_val)
            output[mask] = F.linear(x[mask], quantized_weight, self.linear.bias)
        return self.relu(self.bn(output)), log_probs, chosen_bits

class IAQ_DeepFFN(nn.Module):
    def __init__(self, bit_choices, input_dim, hidden_dim=128, output_dim=10, num_hidden_layers=14):
        super().__init__(); self.layers = nn.ModuleList([QuantizedBlock(input_dim if i==0 else hidden_dim, hidden_dim, bit_choices) for i in range(num_hidden_layers)])
        self.output_layer = nn.Linear(hidden_dim, output_dim); self.apply(_init_weights)
    def forward(self, x, temp=1.0, is_eval=False):
        x = x.view(x.size(0), -1); batch_log_probs, batch_bits = [], []
        for layer in self.layers:
            x, log_p, bits = layer(x, temp, is_eval);
            if log_p is not None: batch_log_probs.append(log_p)
            batch_bits.append(bits)
        output = self.output_layer(x)
        final_log_probs = torch.stack(batch_log_probs, dim=1).sum(dim=1) if batch_log_probs and not is_eval else None
        final_bits = torch.stack(batch_bits, dim=1); return output, final_log_probs, final_bits

# --- Training and Testing Loops ---
def train(model, loader, optimizer, epoch, model_type, stats_log, **kwargs):
    model.train(); pbar = tqdm(loader, desc=f"{C.CYAN}Epoch {epoch} [Train {model_type}]{C.END}", unit="batch")
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(CONFIG['device']), target.to(CONFIG['device'])
        
        if model_type == 'IAQ':
            optim_policy, temp_schedule = kwargs['optim_policy'], kwargs['temp_schedule']
            global_step = (epoch - 1) * len(loader) + batch_idx; temp = temp_schedule[global_step]

            student_logits, log_probs, bits_per_sample = model(data, temp=temp)
            task_loss_per_sample = F.cross_entropy(student_logits, target, reduction='none')
            (grad_per_sample,) = torch.autograd.grad(outputs=task_loss_per_sample.sum(), inputs=student_logits, retain_graph=True, create_graph=False)
            sensitivity_score = torch.norm(grad_per_sample, p=2, dim=1)
            
            # --- NEW: Diversity-Driven Reward Calculation ---
            # 1. Calculate per-layer diversity bonus (standard deviation of choices in the batch)
            # bits_per_sample is [batch_size, num_layers]
            diversity_per_layer = torch.std(bits_per_sample.float(), dim=0) # Result is [num_layers]
            total_diversity_bonus = diversity_per_layer.mean() # Aggregate to a single scalar bonus

            # 2. Formulate the final reward including the diversity bonus
            total_bits = bits_per_sample.sum(dim=1); bit_cost = CONFIG['lambda_bits'] * total_bits
            reward = -task_loss_per_sample.detach() - bit_cost - CONFIG['gamma_sensitivity'] * sensitivity_score.detach()
            
            # Add the diversity bonus to the reward for every sample in the batch
            # This incentivizes the entire policy to encourage exploration
            reward += CONFIG['eta_diversity'] * total_diversity_bonus.detach()

            # 3. Calculate losses and update
            policy_loss = (-log_probs * reward).mean()
            main_task_loss = task_loss_per_sample.mean()
            total_loss = main_task_loss + policy_loss
            optimizer.zero_grad(); optim_policy.zero_grad()
            total_loss.backward()
            optimizer.step(); optim_policy.step()

            # Logging
            stats_log['rewards'].append(reward.mean().item()); stats_log['avg_bits'].append(total_bits.mean().item())
            pbar.set_postfix(loss=f"{main_task_loss.item():.3f}", reward=f"{reward.mean():.2f}", bits=f"{total_bits.mean().item():.1f}", diversity=f"{total_diversity_bonus.item():.3f}")
            stats_log['train_loss'].append(main_task_loss.item())
        else:
            optimizer.zero_grad(); output = model(data); loss = F.cross_entropy(output, target)
            loss.backward(); optimizer.step()
            pbar.set_postfix(loss=f"{loss.item():.4f}"); stats_log['train_loss'].append(loss.item())

def test(model, loader, epoch, model_type, stats_log):
    model.eval(); test_loss = 0; correct = 0
    pbar = tqdm(loader, desc=f"{C.YELLOW}Epoch {epoch} [Test {model_type}]{C.END}", unit="batch")
    all_bits, all_losses = [], []
    with torch.no_grad():
        for data, target in pbar:
            data, target = data.to(CONFIG['device']), target.to(CONFIG['device'])
            if model_type == 'IAQ':
                output, _, bits_per_sample = model(data, is_eval=True)
                total_bits_per_sample = bits_per_sample.sum(dim=1); all_bits.extend(total_bits_per_sample.cpu().tolist())
                all_losses.extend(F.cross_entropy(output, target, reduction='none').cpu().tolist())
                if 'final_bit_decisions' not in stats_log: stats_log['final_bit_decisions'] = defaultdict(list)
                for i in range(bits_per_sample.shape[1]): stats_log['final_bit_decisions'][i].extend(bits_per_sample[:, i].cpu().tolist())
            else: output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item(); correct += output.argmax(dim=1).eq(target).sum().item()
    test_loss /= len(loader.dataset); accuracy = 100. * correct / len(loader.dataset)
    stats_log['test_loss'].append(test_loss); stats_log['test_acc'].append(accuracy)
    avg_bits_str = f", Avg Bits: {np.mean(all_bits):.2f}" if model_type == 'IAQ' and all_bits else ""
    print(f"  {C.YELLOW}└> Test Results: Avg loss: {test_loss:.4f}, Accuracy: {accuracy:.2f}%{avg_bits_str}{C.END}")
    if model_type == 'IAQ': stats_log['final_test_bits_vs_loss'] = (all_losses, all_bits)

# --- Analysis & Visualization ---
def calculate_model_stats(model, model_type, iaq_stats=None):
    stats = {}; params_backbone = sum(p.numel() for n, p in model.named_parameters() if 'controller' not in n)
    if model_type == 'Naive': stats['params'], stats['size_mb'] = params_backbone, params_backbone * 4 / (1024**2)
    elif model_type == 'QAT': stats['params'], stats['size_mb'] = params_backbone, params_backbone * model.static_bits / (8 * 1024**2)
    else:
        params_controllers = sum(p.numel() for n, p in model.named_parameters() if 'controller' in n); stats['params'] = params_backbone + params_controllers
        avg_bits_per_weight = np.mean(list(iaq_stats['final_bit_decisions'].values())) if iaq_stats and iaq_stats['final_bit_decisions'] else 8.0
        stats['size_mb'] = (params_backbone * avg_bits_per_weight + params_controllers * 32) / (8 * 1024**2)
    stats['gflops'] = sum(2 * m.in_features * m.out_features for m in model.modules() if isinstance(m, nn.Linear)) / 1e9; return stats

def create_visualizations(all_stats):
    fig, axes = plt.subplots(2, 2, figsize=(18, 14)); fig.suptitle('Diversity-Driven IAQ on CIFAR-10 vs. Baselines', fontsize=22, weight='bold')
    colors = {'Naive': 'blue', 'QAT': 'orange', 'IAQ': 'green'}; markers = {'Naive': 'o', 'QAT': 's', 'IAQ': 'x'}
    for name, stats in all_stats.items():
        axes[0,0].plot(stats['test_acc'], label=f"{name} (Final: {stats['test_acc'][-1]:.2f}%)", color=colors[name], marker=markers[name], alpha=0.8)
        axes[0,1].plot(stats['test_loss'], label=name, color=colors[name], marker=markers[name], alpha=0.8)
    axes[0,0].set_title("Test Accuracy Comparison"); axes[0,0].set_xlabel("Epochs"); axes[0,0].set_ylabel("Accuracy (%)"); axes[0,0].grid(True); axes[0,0].legend()
    axes[0,1].set_title("Test Loss Comparison"); axes[0,1].set_xlabel("Epochs"); axes[0,1].set_ylabel("Cross-Entropy Loss"); axes[0,1].grid(True); axes[0,1].legend()
    if 'IAQ' in all_stats:
        iaq_stats = all_stats['IAQ']
        if iaq_stats.get('final_bit_decisions'):
            df = pd.DataFrame(iaq_stats['final_bit_decisions']); df.columns = [f'L{i+1}' for i in df.columns]
            sns.countplot(data=df.melt(var_name='Layer', value_name='Bits'), x='Layer', hue='Bits', ax=axes[1,0], palette='viridis')
            axes[1,0].set_title("IAQ: Final Bitwidth Choices per Layer (SHOULD SHOW DIVERSITY)"); axes[1,0].set_ylabel("Count"); axes[1,0].tick_params(axis='x', rotation=45)
        losses, bits = iaq_stats.get('final_test_bits_vs_loss', ([], []))
        if losses and bits:
            sns.regplot(x=losses, y=bits, ax=axes[1,1], scatter_kws={'alpha':0.1, 's':10}, line_kws={'color':'red'})
            axes[1,1].set_title("IAQ: Bit Usage vs. Input 'Difficulty'"); axes[1,1].set_xlabel("Per-Sample Loss"); axes[1,1].set_ylabel("Total Bits Used"); axes[1,1].grid(True)
    plt.tight_layout(rect=[0, 0, 1, 0.96]); plt.savefig(CONFIG['output_filename'], dpi=120)
    print(f"\n{C.GREEN}Visualizations saved to {C.BOLD}{CONFIG['output_filename']}{C.END}"); plt.close()

# --- Main Execution ---
if __name__ == '__main__':
    print(f"{C.HEADER}{C.BOLD}Starting Diversity-Driven IAQ Experiment on CIFAR-10{C.END}")
    print(f"Using device: {C.BOLD}{CONFIG['device']}{C.END}\n")
    all_stats_logs = defaultdict(lambda: defaultdict(list))
    models = {}
    input_dim = CONFIG['input_dim']
    models['Naive'] = DeepFFN(input_dim=input_dim, hidden_dim=CONFIG['hidden_dim'], num_hidden_layers=CONFIG['num_hidden_layers']).to(CONFIG['device'])
    models['QAT'] = QAT_DeepFFN(input_dim=input_dim, hidden_dim=CONFIG['hidden_dim'], num_hidden_layers=CONFIG['num_hidden_layers'], static_bits=4).to(CONFIG['device'])
    models['IAQ'] = IAQ_DeepFFN(bit_choices=CONFIG['bit_choices'], input_dim=input_dim, hidden_dim=CONFIG['hidden_dim'], num_hidden_layers=CONFIG['num_hidden_layers']).to(CONFIG['device'])
    for name, model in models.items():
        print(f"{C.BLUE}--- Training {name} Model ---{C.END}")
        train_loader, test_loader = get_data_loaders(CONFIG['batch_size'])
        if name == 'IAQ':
            main_params = [p for n, p in model.named_parameters() if 'controller' not in n]
            policy_params = [p for n, p in model.named_parameters() if 'controller' in n]
            optimizer = torch.optim.Adam(main_params, lr=CONFIG['lr'], weight_decay=1e-4)
            optim_policy = torch.optim.Adam(policy_params, lr=CONFIG['lr_policy'])
            temp_schedule = np.linspace(CONFIG['gumbel_temp_initial'], CONFIG['gumbel_temp_final'], len(train_loader) * CONFIG['epochs'])
            train_kwargs = {'optim_policy': optim_policy, 'temp_schedule': temp_schedule}
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['lr'], weight_decay=1e-4)
            train_kwargs = {}
        for epoch in range(1, CONFIG['epochs'] + 1):
            train(model, train_loader, optimizer, epoch, name, all_stats_logs[name], **train_kwargs)
            test(model, test_loader, epoch, name, all_stats_logs[name])
        print("\n")
    print(f"\n{C.HEADER}{C.BOLD}--- Final Performance & Cost Analysis (CIFAR-10) ---{C.END}")
    perf_stats_list = []
    for name, model in models.items():
        stats = calculate_model_stats(model, name, all_stats_logs[name])
        stats['Accuracy'] = f"{all_stats_logs[name]['test_acc'][-1]:.2f}%"; stats['Model'] = f"{name} FFN"
        perf_stats_list.append(stats)
    df = pd.DataFrame(perf_stats_list).set_index('Model')
    df_display = df[['Accuracy', 'params', 'size_mb', 'gflops']].copy()
    df_display.rename(columns={'params': 'Params', 'size_mb': 'Effective Size (MB)', 'gflops': 'GFLOPs'}, inplace=True)
    print(df_display.to_string(formatters={'Effective Size (MB)': '{:.2f}'.format, 'GFLOPs': '{:.3f}'.format}))
    create_visualizations(all_stats_logs)
