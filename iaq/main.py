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
from PIL import Image

# --- Configuration ---
CONFIG = {
    "epochs": 5,
    "batch_size": 128,
    "lr_main": 1e-3,
    "lr_policy": 5e-4, # Separate learning rate for controllers
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "log_interval": 200,
    "lambda_bits": 0.001, # Penalty for bit usage in the reward
    "bit_choices": [2, 4, 8],
    "gumbel_temp_initial": 5.0,
    "gumbel_temp_final": 0.5,
}

# --- Data Loading ---
def get_data_loaders(batch_size):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

# --- Naive FFN Model ---
class NaiveFFN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(784, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 10)
        )

    def forward(self, x):
        return self.layers(x.view(-1, 784))

# --- IAQ Model Components ---

# 1. Fake Quantization with Straight-Through Estimator (STE)
class FakeQuantize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, bits):
        if bits == 32: # No quantization
            return x
        # Scale to [0, 2^bits - 1]
        q_level = 2**bits - 1
        min_val, max_val = x.min(), x.max()
        scale = (max_val - min_val) / q_level
        zero_point = min_val
        # Quantize and de-quantize
        quantized = torch.round((x - zero_point) / (scale + 1e-8))
        dequantized = quantized * scale + zero_point
        return dequantized

    @staticmethod
    def backward(ctx, grad_output):
        # STE: Pass gradient through unchanged
        return grad_output, None

apply_fake_quant = FakeQuantize.apply

# 2. Quantization Controller
class QuantizationController(nn.Module):
    def __init__(self, num_choices):
        super().__init__()
        self.controller_net = nn.Sequential(
            nn.Linear(3, 16), # Input: mean, var, sparsity
            nn.ReLU(),
            nn.Linear(16, num_choices)
        )
    def forward(self, x):
        return self.controller_net(x)

# 3. Dynamically Quantized Layer
class QuantizedLayer(nn.Module):
    def __init__(self, in_features, out_features, bit_choices):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.relu = nn.ReLU()
        self.bit_choices = bit_choices
        self.controller = QuantizationController(len(bit_choices))

    def forward(self, x, temp, is_eval=False):
        # 1. Calculate activation statistics
        with torch.no_grad():
            act_mean = x.mean()
            act_var = x.var()
            act_sparsity = (x == 0).float().mean()
            stats = torch.tensor([act_mean, act_var, act_sparsity], device=x.device)

        # 2. Controller decides on bitwidth
        logits = self.controller(stats)

        # 3. Sample bitwidth (Gumbel-Softmax for training, Argmax for eval)
        if is_eval:
            choice_idx = logits.argmax()
            log_prob = 0 # Not needed for eval
        else:
            gumbel_probs = F.gumbel_softmax(logits.unsqueeze(0), tau=temp, hard=True).squeeze(0)
            choice_idx = gumbel_probs.argmax()
            log_prob = torch.log(F.softmax(logits, dim=-1)[choice_idx] + 1e-9)
        
        chosen_bits = self.bit_choices[choice_idx]
        
        # 4. Apply fake quantization to weights
        quantized_weight = apply_fake_quant(self.linear.weight, float(chosen_bits))
        
        # 5. Forward pass with quantized weights
        output = F.linear(x, quantized_weight, self.linear.bias)
        output = self.relu(output)
        
        return output, log_prob, chosen_bits

# 4. Full IAQ FFN Model
class IAQ_FFN(nn.Module):
    def __init__(self, bit_choices):
        super().__init__()
        self.quant_layer1 = QuantizedLayer(784, 256, bit_choices)
        self.quant_layer2 = QuantizedLayer(256, 128, bit_choices)
        self.quant_layer3 = QuantizedLayer(128, 64, bit_choices)
        self.quant_layer4 = QuantizedLayer(64, 32, bit_choices)
        self.output_layer = nn.Linear(32, 10)

    def forward(self, x, temp=1.0, is_eval=False):
        x = x.view(-1, 784)
        log_probs_list, bits_list = [], []

        x, log_p, bits = self.quant_layer1(x, temp, is_eval)
        log_probs_list.append(log_p)
        bits_list.append(bits)

        x, log_p, bits = self.quant_layer2(x, temp, is_eval)
        log_probs_list.append(log_p)
        bits_list.append(bits)

        x, log_p, bits = self.quant_layer3(x, temp, is_eval)
        log_probs_list.append(log_p)
        bits_list.append(bits)

        x, log_p, bits = self.quant_layer4(x, temp, is_eval)
        log_probs_list.append(log_p)
        bits_list.append(bits)
        
        output = self.output_layer(x)
        return output, log_probs_list, bits_list

# --- Training and Evaluation Loops ---

def train_naive(model, device, train_loader, optimizer, epoch, stats):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
    stats['train_loss'].append(loss.item())

def test_naive(model, device, test_loader, stats):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    stats['test_loss'].append(test_loss)
    stats['test_acc'].append(accuracy)
    print(f'Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)')

def train_iaq(model, device, train_loader, optim_main, optim_policy, epoch, stats):
    model.train()
    temp_schedule = np.linspace(CONFIG['gumbel_temp_initial'], CONFIG['gumbel_temp_final'], len(train_loader) * CONFIG['epochs'])
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optim_main.zero_grad()
        optim_policy.zero_grad()
        
        batch_log_probs = []
        batch_bit_costs = []
        batch_task_losses = []

        # Process one sample at a time for RL
        for i in range(data.size(0)):
            sample, sample_target = data[i:i+1], target[i:i+1]
            current_iter = (epoch - 1) * len(train_loader) + batch_idx
            temp = temp_schedule[current_iter]

            output, log_probs, bits = model(sample, temp=temp)
            
            # 1. Task Loss (for main network)
            task_loss = F.cross_entropy(output, sample_target)
            
            # 2. Bit Cost & Reward (for controllers)
            total_bits = sum(bits)
            bit_cost = CONFIG['lambda_bits'] * total_bits
            reward = -task_loss.detach() - bit_cost # Use detached loss for reward signal
            
            # 3. Policy Loss (REINFORCE)
            policy_loss = -sum(log_probs) * reward
            
            # Accumulate losses for the batch
            # We backpropagate the combined loss for stability
            total_loss = task_loss + policy_loss
            total_loss.backward()

            # Log stats
            batch_log_probs.extend(log_probs)
            batch_bit_costs.append(total_bits)
            batch_task_losses.append(task_loss.item())
        
        stats['rewards'].append(reward.item())
        stats['avg_bits'].append(np.mean(batch_bit_costs))
        
        # Step optimizers after processing the batch
        optim_main.step()
        optim_policy.step()

    stats['train_loss'].append(np.mean(batch_task_losses))

def test_iaq(model, device, test_loader, epoch, stats):
    model.eval()
    test_loss = 0
    correct = 0
    # For detailed analysis plots
    all_bits = []
    all_losses = []
    all_bit_decisions = defaultdict(list)

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            for i in range(data.size(0)):
                sample, sample_target = data[i:i+1], target[i:i+1]
                output, _, bits = model(sample, is_eval=True)
                
                loss = F.cross_entropy(output, sample_target)
                test_loss += loss.item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(sample_target.view_as(pred)).sum().item()
                
                # Store data for plots
                all_bits.append(sum(bits))
                all_losses.append(loss.item())
                for layer_idx, bit_val in enumerate(bits):
                    all_bit_decisions[layer_idx].append(bit_val)

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    avg_total_bits = np.mean(all_bits)

    stats['test_loss'].append(test_loss)
    stats['test_acc'].append(accuracy)
    stats['final_test_bits_vs_loss'] = (all_losses, all_bits)
    stats['final_bit_decisions'] = all_bit_decisions

    print(f'Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%), Avg Bits: {avg_total_bits:.2f}')


def create_visualizations(naive_stats, iaq_stats):
    fig, axes = plt.subplots(3, 2, figsize=(16, 20))
    fig.suptitle('Naive FFN vs. Input-Adaptive Quantization (IAQ) FFN', fontsize=20)

    # 1. Accuracy Comparison
    ax = axes[0, 0]
    ax.plot(naive_stats['test_acc'], label=f"Naive (Final: {naive_stats['test_acc'][-1]:.2f}%)", marker='o')
    ax.plot(iaq_stats['test_acc'], label=f"IAQ (Final: {iaq_stats['test_acc'][-1]:.2f}%)", marker='x')
    ax.set_title("Test Accuracy Comparison")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Accuracy (%)")
    ax.grid(True)
    ax.legend()

    # 2. Loss Comparison
    ax = axes[0, 1]
    ax.plot(naive_stats['test_loss'], label="Naive", marker='o')
    ax.plot(iaq_stats['test_loss'], label="IAQ", marker='x')
    ax.set_title("Test Loss Comparison")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Cross-Entropy Loss")
    ax.grid(True)
    ax.legend()
    
    # 3. IAQ Training Dynamics (Reward & Bits)
    ax = axes[1, 0]
    ax2 = ax.twinx()
    ax.plot(iaq_stats['avg_bits'], 'g-', label='Avg Total Bits per Sample')
    ax2.plot(iaq_stats['rewards'], 'b-', label='Reward (Sampled)', alpha=0.6)
    ax.set_xlabel("Training Steps")
    ax.set_ylabel("Average Bits", color='g')
    ax2.set_ylabel("Reward", color='b')
    ax.set_title("IAQ: Reward and Bit Usage During Training")
    ax.legend(loc='upper left')
    ax2.legend(loc='upper right')
    ax.grid(True)
    
    # 4. Bitwidth vs. Input Difficulty (Loss)
    ax = axes[1, 1]
    losses, bits = iaq_stats['final_test_bits_vs_loss']
    sns.regplot(x=losses, y=bits, ax=ax, scatter_kws={'alpha':0.2}, line_kws={'color':'red'})
    ax.set_title("Bit Usage vs. Input 'Difficulty' (Loss)")
    ax.set_xlabel("Per-Sample Loss (Higher = Harder)")
    ax.set_ylabel("Total Bits Used")
    ax.grid(True)
    
    # 5. Bitwidth Distribution per Layer
    ax = axes[2, 0]
    bit_decisions = iaq_stats['final_bit_decisions']
    df = pd.DataFrame(bit_decisions)
    df.columns = [f'Layer {i+1}' for i in df.columns]
    df_melted = df.melt(var_name='Layer', value_name='Bits')
    sns.countplot(data=df_melted, x='Layer', hue='Bits', ax=ax, palette='viridis')
    ax.set_title("Bitwidth Choices per Layer on Test Set")
    ax.set_ylabel("Count")
    
    # 6. Heatmap of Per-Input Bit Allocation
    ax = axes[2, 1]
    num_samples_to_show = 25
    sample_decisions = pd.DataFrame(bit_decisions).iloc[:num_samples_to_show]
    sns.heatmap(sample_decisions, ax=ax, cmap='viridis', cbar_kws={'label': 'Chosen Bitwidth'})
    ax.set_title(f"Bit Allocation for First {num_samples_to_show} Test Samples")
    ax.set_xlabel("Layer Index")
    ax.set_ylabel("Test Sample Index")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save to an in-memory buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100)
    buf.seek(0)
    img = Image.open(buf)
    return img

# --- Main Execution ---
if __name__ == '__main__':
    try:
        import pandas as pd
    except ImportError:
        print("Pandas not found. Installing...")
        import subprocess
        import sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pandas"])
        import pandas as pd

    # --- Naive Model Run ---
    print("--- 1. Training Naive Baseline Model ---")
    naive_stats = defaultdict(list)
    train_loader, test_loader = get_data_loaders(CONFIG['batch_size'])
    naive_model = NaiveFFN().to(CONFIG['device'])
    optimizer = torch.optim.Adam(naive_model.parameters(), lr=CONFIG['lr_main'])
    
    param_count = sum(p.numel() for p in naive_model.parameters() if p.requires_grad)
    print(f"Naive model has {param_count:,} parameters.")

    for epoch in range(1, CONFIG['epochs'] + 1):
        print(f"Epoch {epoch}/{CONFIG['epochs']}")
        train_naive(naive_model, CONFIG['device'], train_loader, optimizer, epoch, naive_stats)
        test_naive(naive_model, CONFIG['device'], test_loader, naive_stats)
    
    # --- IAQ Model Run ---
    print("\n\n--- 2. Training Input-Adaptive Quantization (IAQ) Model ---")
    iaq_stats = defaultdict(list)
    train_loader, test_loader = get_data_loaders(1) # RL part works best with batch size 1
    iaq_model = IAQ_FFN(bit_choices=CONFIG['bit_choices']).to(CONFIG['device'])

    main_params = [p for name, p in iaq_model.named_parameters() if 'controller' not in name]
    policy_params = [p for name, p in iaq_model.named_parameters() if 'controller' in name]
    
    optim_main = torch.optim.Adam(main_params, lr=CONFIG['lr_main'])
    optim_policy = torch.optim.Adam(policy_params, lr=CONFIG['lr_policy'])
    
    param_count = sum(p.numel() for p in iaq_model.parameters() if p.requires_grad)
    print(f"IAQ model has {param_count:,} parameters (includes controllers).")
    
    for epoch in range(1, CONFIG['epochs'] + 1):
        print(f"Epoch {epoch}/{CONFIG['epochs']}")
        train_iaq(iaq_model, CONFIG['device'], train_loader, optim_main, optim_policy, epoch, iaq_stats)
        test_iaq(iaq_model, CONFIG['device'], test_loader, epoch, iaq_stats)

    # --- Generate and Show Final Report ---
    print("\n\n--- 3. Generating Final Report ---")
    final_image = create_visualizations(naive_stats, iaq_stats)
    print("Analysis complete. Displaying results...")
    final_image.show()
