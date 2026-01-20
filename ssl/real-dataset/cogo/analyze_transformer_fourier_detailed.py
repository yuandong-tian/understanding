"""
Detailed Fourier analysis for Transformer models.

This script provides deeper analysis of where Fourier weight structures emerge
in transformers, with specific focus on:
1. Per-attention-head analysis
2. QK vs V projection differences
3. FFN layer frequency patterns
4. Comparison with theoretical MLP Fourier patterns

Usage:
    python analyze_transformer_fourier_detailed.py <model_folder> [options]
"""

import torch
import torch.nn as nn
import math
import os
import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from analyze_util import construct_bases
from visualize_transformer_fourier import (
    get_fourier_basis,
    extract_transformer_weights,
    extract_mlp_weights,
)


# ============================================================================
# Detailed Attention Analysis
# ============================================================================

def split_qkv_weights(in_proj_weight, hidden_size):
    """
    Split combined QKV projection weights.

    in_proj_weight: [3*hidden_size, hidden_size]
    Returns: Q, K, V each of shape [hidden_size, hidden_size]
    """
    assert in_proj_weight.shape[0] == 3 * hidden_size
    Q = in_proj_weight[:hidden_size, :]
    K = in_proj_weight[hidden_size:2*hidden_size, :]
    V = in_proj_weight[2*hidden_size:, :]
    return Q, K, V


def split_into_heads(weight, n_heads):
    """
    Split weight matrix into per-head components.

    weight: [hidden_size, hidden_size]
    Returns: [n_heads, head_dim, hidden_size]
    """
    hidden_size = weight.shape[0]
    head_dim = hidden_size // n_heads
    return weight.view(n_heads, head_dim, -1)


def analyze_attention_head_fourier(attn_weights, M, hidden_size, n_heads):
    """
    Analyze Fourier structure per attention head.

    Returns per-head analysis including:
    - QK pattern (attention pattern structure)
    - V projection structure
    - Output projection structure
    """
    fourier_basis = get_fourier_basis(M)
    head_dim = hidden_size // n_heads

    results = {}

    for layer_idx, layer_weights in attn_weights.items():
        layer_results = {
            'heads': defaultdict(dict),
            'aggregate': {},
        }

        # Get Q, K, V, O projections
        if 'in_proj' in layer_weights:
            Q, K, V = split_qkv_weights(layer_weights['in_proj'], hidden_size)
        else:
            Q = layer_weights.get('q_proj')
            K = layer_weights.get('k_proj')
            V = layer_weights.get('v_proj')

        O = layer_weights.get('out_proj')

        # Split into heads and analyze each
        if Q is not None:
            Q_heads = split_into_heads(Q, n_heads)
            K_heads = split_into_heads(K, n_heads)
            V_heads = split_into_heads(V, n_heads)

            for h in range(n_heads):
                Qh = Q_heads[h]  # [head_dim, hidden_size]
                Kh = K_heads[h]
                Vh = V_heads[h]

                # Compute QK^T pattern (what attention sees)
                QK_pattern = Qh @ Kh.t()  # [head_dim, head_dim]

                # SVD of QK pattern
                U, S, Vt = torch.linalg.svd(QK_pattern.cfloat(), full_matrices=False)

                layer_results['heads'][h]['QK_singular_values'] = S
                layer_results['heads'][h]['QK_rank_90'] = (S.cumsum(0) / S.sum() < 0.9).sum().item() + 1

                # Analyze V projection
                U_v, S_v, Vt_v = torch.linalg.svd(Vh.cfloat(), full_matrices=False)
                layer_results['heads'][h]['V_singular_values'] = S_v
                layer_results['heads'][h]['V_rank_90'] = (S_v.cumsum(0) / S_v.sum() < 0.9).sum().item() + 1

                # If hidden_size == M, check Fourier structure
                if hidden_size == M:
                    # Fourier analysis of QK pattern
                    QK_fourier = fourier_basis.conj().t() @ QK_pattern.cfloat() @ fourier_basis / M
                    layer_results['heads'][h]['QK_fourier_diag'] = torch.diag(QK_fourier).abs()

            # Aggregate statistics across heads
            all_QK_ranks = [layer_results['heads'][h]['QK_rank_90'] for h in range(n_heads)]
            all_V_ranks = [layer_results['heads'][h]['V_rank_90'] for h in range(n_heads)]

            layer_results['aggregate']['mean_QK_rank'] = np.mean(all_QK_ranks)
            layer_results['aggregate']['mean_V_rank'] = np.mean(all_V_ranks)

        # Analyze output projection
        if O is not None:
            U_o, S_o, Vt_o = torch.linalg.svd(O.cfloat(), full_matrices=False)
            layer_results['aggregate']['O_singular_values'] = S_o
            layer_results['aggregate']['O_rank_90'] = (S_o.cumsum(0) / S_o.sum() < 0.9).sum().item() + 1

        results[layer_idx] = layer_results

    return results


def analyze_ffn_fourier_detailed(ffn_weights, M, hidden_size):
    """
    Detailed FFN Fourier analysis.

    FFN: x -> W1 -> act -> W2 -> output
    Effective: W2 @ diag(act(W1 @ x))

    For square activation, this creates patterns similar to MLP.
    """
    fourier_basis = get_fourier_basis(M)
    results = {}

    for layer_idx, layer_weights in ffn_weights.items():
        W1 = layer_weights.get('linear1')  # [dim_ffn, hidden_size]
        W2 = layer_weights.get('linear2')  # [hidden_size, dim_ffn]

        if W1 is None or W2 is None:
            continue

        layer_results = {}

        # Analyze W1
        W1_c = W1.cfloat()
        U1, S1, V1t = torch.linalg.svd(W1_c, full_matrices=False)
        layer_results['W1'] = {
            'shape': W1.shape,
            'singular_values': S1,
            'rank_90': (S1.cumsum(0) / S1.sum() < 0.9).sum().item() + 1,
            'condition_number': (S1[0] / S1[-1]).item() if S1[-1] > 1e-10 else float('inf'),
        }

        # Analyze W2
        W2_c = W2.cfloat()
        U2, S2, V2t = torch.linalg.svd(W2_c, full_matrices=False)
        layer_results['W2'] = {
            'shape': W2.shape,
            'singular_values': S2,
            'rank_90': (S2.cumsum(0) / S2.sum() < 0.9).sum().item() + 1,
            'condition_number': (S2[0] / S2[-1]).item() if S2[-1] > 1e-10 else float('inf'),
        }

        # Combined W2 @ W1 analysis (like linear skip connection)
        W_combined = W2_c @ W1_c
        U_c, S_c, V_ct = torch.linalg.svd(W_combined, full_matrices=False)
        layer_results['W_combined'] = {
            'shape': W_combined.shape,
            'singular_values': S_c,
            'rank_90': (S_c.cumsum(0) / S_c.sum() < 0.9).sum().item() + 1,
        }

        # If hidden_size == M, do Fourier analysis
        if hidden_size == M:
            # Fourier analysis of combined transformation
            W_fourier = fourier_basis.conj().t() @ W_combined @ fourier_basis / M
            layer_results['W_combined_fourier_diag'] = torch.diag(W_fourier).abs()

            # Check how diagonal the Fourier representation is
            diag_power = W_fourier.diag().abs().pow(2).sum()
            total_power = W_fourier.abs().pow(2).sum()
            layer_results['fourier_diagonality'] = (diag_power / total_power).item()

        results[layer_idx] = layer_results

    return results


# ============================================================================
# Comparison with Theoretical MLP Patterns
# ============================================================================

def compute_mlp_theoretical_pattern(M, k):
    """
    Compute the theoretical Fourier pattern for MLP with quadratic activation.

    For modular addition with quadratic activation:
    - W should have rows proportional to Fourier basis vectors
    - Pattern: W[j,:] ∝ e^{2πi k j / M} for some frequency k
    """
    fourier_basis = get_fourier_basis(M)

    # Theoretical pattern: each hidden unit responds to one frequency
    # W[:, j] = a * fourier_basis[:, k] for some a and k
    # V[j, :] = b * fourier_basis[:, k].conj() for conjugate frequency

    return fourier_basis[:, k]


def compare_with_mlp_pattern(transformer_weights, M, hidden_size):
    """
    Compare transformer weight patterns with theoretical MLP Fourier patterns.

    Returns similarity scores for different frequency patterns.
    """
    fourier_basis = get_fourier_basis(M)
    results = {}

    # Get embedding layer
    emb = transformer_weights.get('embedding')
    if emb is not None and emb.shape[0] == M:
        emb_c = emb.cfloat()

        # Project to Fourier domain
        emb_fourier = fourier_basis.conj().t() @ emb_c / M

        # For each hidden dimension, find the dominant frequency
        dominant_freqs = emb_fourier.abs().argmax(dim=0)
        freq_counts = torch.bincount(dominant_freqs, minlength=M)

        results['embedding'] = {
            'dominant_freq_distribution': freq_counts,
            'unique_dominant_freqs': (freq_counts > 0).sum().item(),
            'max_freq_count': freq_counts.max().item(),
        }

        # Compute how "MLP-like" the embedding is
        # In ideal MLP, each hidden unit has ONE dominant frequency
        emb_fourier_norm = emb_fourier.abs()
        max_per_hidden = emb_fourier_norm.max(dim=0).values
        total_per_hidden = emb_fourier_norm.sum(dim=0)
        sparsity = (max_per_hidden / (total_per_hidden + 1e-10)).mean().item()

        results['embedding']['mlp_similarity'] = sparsity

    # Get output V
    V = transformer_weights.get('output_V')
    if V is not None and V.shape[0] == M:
        V_c = V.cfloat()
        # V shape is [M, hidden_size], project M dimension to Fourier
        V_fourier = fourier_basis.conj().t() @ V_c / M

        # Similar analysis for V
        dominant_freqs = V_fourier.abs().argmax(dim=0)  # For each hidden unit
        freq_counts = torch.bincount(dominant_freqs, minlength=M)

        results['output_V'] = {
            'dominant_freq_distribution': freq_counts,
            'unique_dominant_freqs': (freq_counts > 0).sum().item(),
        }

        V_fourier_norm = V_fourier.abs()
        max_per_hidden = V_fourier_norm.max(dim=0).values
        total_per_hidden = V_fourier_norm.sum(dim=0)
        sparsity = (max_per_hidden / (total_per_hidden + 1e-10)).mean().item()

        results['output_V']['mlp_similarity'] = sparsity

    return results


# ============================================================================
# Location Analysis: Where are Fourier Weights?
# ============================================================================

def locate_fourier_structures(model_path, M=None, threshold=0.2):
    """
    Identify which parts of the transformer have strong Fourier structure.

    Returns a report of where Fourier patterns are strongest.
    """
    data = torch.load(model_path, map_location='cpu')
    state_dict = data['model']

    # Infer M
    if M is None:
        if 'embedding.weight' in state_dict:
            M = state_dict['embedding.weight'].shape[0]
        elif 'V.weight' in state_dict:
            M = state_dict['V.weight'].shape[0]

    fourier_basis = get_fourier_basis(M)

    report = {
        'M': M,
        'strong_fourier_locations': [],
        'weak_fourier_locations': [],
        'layer_scores': {},
    }

    def compute_fourier_score(weight, name):
        """Compute how 'Fourier-like' a weight matrix is."""
        weight_c = weight.cfloat()

        # Try projecting from both dimensions
        scores = []

        if weight.shape[0] == M:
            # Project the first dimension (output) to Fourier domain
            # weight: [M, D], fourier_basis: [M, M]
            # Result: [M, D] in Fourier domain
            proj = fourier_basis.conj().t() @ weight_c / M
            # Compute diagonality in Fourier domain
            if proj.shape[0] == proj.shape[1]:
                diag_power = proj.diag().abs().pow(2).sum()
                total_power = proj.abs().pow(2).sum()
                scores.append(('output_dim', (diag_power / total_power).item()))

            # Compute sparsity
            norm_proj = proj.abs()
            max_vals = norm_proj.max(dim=0).values
            sum_vals = norm_proj.sum(dim=0)
            sparsity = (max_vals / (sum_vals + 1e-10)).mean().item()
            scores.append(('output_sparsity', sparsity))

        if weight.shape[1] == M:
            # Project the second dimension (input) to Fourier domain
            # weight: [D, M], fourier_basis: [M, M]
            # Result: [D, M] in Fourier domain
            proj = weight_c @ fourier_basis.conj() / M
            norm_proj = proj.abs()
            max_vals = norm_proj.max(dim=1).values
            sum_vals = norm_proj.sum(dim=1)
            sparsity = (max_vals / (sum_vals + 1e-10)).mean().item()
            scores.append(('input_sparsity', sparsity))

        return scores

    # Analyze each weight
    for key, weight in state_dict.items():
        if 'weight' not in key:
            continue

        if weight.dim() != 2:
            continue

        scores = compute_fourier_score(weight, key)

        if scores:
            best_score = max(s[1] for s in scores)
            report['layer_scores'][key] = {
                'shape': weight.shape,
                'scores': dict(scores),
                'best_score': best_score,
            }

            if best_score > threshold:
                report['strong_fourier_locations'].append((key, best_score))
            else:
                report['weak_fourier_locations'].append((key, best_score))

    # Sort by score
    report['strong_fourier_locations'].sort(key=lambda x: -x[1])
    report['weak_fourier_locations'].sort(key=lambda x: -x[1])

    return report


# ============================================================================
# Visualization
# ============================================================================

def plot_attention_head_analysis(attn_analysis, save_path=None):
    """Plot per-head attention analysis."""
    n_layers = len(attn_analysis)

    fig, axes = plt.subplots(n_layers, 3, figsize=(15, 4 * n_layers))
    if n_layers == 1:
        axes = axes.reshape(1, -1)

    for layer_idx, layer_results in attn_analysis.items():
        row = layer_idx

        # Plot QK singular values for each head
        ax1 = axes[row, 0]
        for h, head_data in layer_results['heads'].items():
            if 'QK_singular_values' in head_data:
                S = head_data['QK_singular_values'][:20].cpu().numpy()
                ax1.semilogy(S, 'o-', label=f'Head {h}', alpha=0.7)
        ax1.set_title(f'Layer {layer_idx}: QK Singular Values')
        ax1.set_xlabel('Index')
        ax1.set_ylabel('Singular Value')
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)

        # Plot V singular values for each head
        ax2 = axes[row, 1]
        for h, head_data in layer_results['heads'].items():
            if 'V_singular_values' in head_data:
                S = head_data['V_singular_values'][:20].cpu().numpy()
                ax2.semilogy(S, 'o-', label=f'Head {h}', alpha=0.7)
        ax2.set_title(f'Layer {layer_idx}: V Projection Singular Values')
        ax2.set_xlabel('Index')
        ax2.set_ylabel('Singular Value')
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)

        # Plot rank summary
        ax3 = axes[row, 2]
        n_heads = len(layer_results['heads'])
        QK_ranks = [layer_results['heads'][h].get('QK_rank_90', 0) for h in range(n_heads)]
        V_ranks = [layer_results['heads'][h].get('V_rank_90', 0) for h in range(n_heads)]

        x = np.arange(n_heads)
        width = 0.35
        ax3.bar(x - width/2, QK_ranks, width, label='QK Rank', alpha=0.7)
        ax3.bar(x + width/2, V_ranks, width, label='V Rank', alpha=0.7)
        ax3.set_title(f'Layer {layer_idx}: Effective Ranks (90% energy)')
        ax3.set_xlabel('Head')
        ax3.set_ylabel('Rank')
        ax3.set_xticks(x)
        ax3.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to: {save_path}")
    else:
        plt.show()

    return fig


def plot_fourier_location_report(report, save_path=None):
    """Plot Fourier structure location report."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Strong locations
    ax1 = axes[0]
    if report['strong_fourier_locations']:
        names = [x[0].replace('transformer_encoder.layers.', 'L').replace('.', '\n')
                 for x in report['strong_fourier_locations'][:15]]
        scores = [x[1] for x in report['strong_fourier_locations'][:15]]
        y_pos = np.arange(len(names))
        ax1.barh(y_pos, scores, color='green', alpha=0.7)
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(names, fontsize=8)
        ax1.set_xlabel('Fourier Score')
        ax1.set_title('Strong Fourier Structures')
        ax1.axvline(0.5, color='red', linestyle='--', alpha=0.5, label='High threshold')
        ax1.legend()

    # Score distribution
    ax2 = axes[1]
    all_scores = [v['best_score'] for v in report['layer_scores'].values()]
    ax2.hist(all_scores, bins=20, alpha=0.7, edgecolor='black')
    ax2.set_xlabel('Fourier Score')
    ax2.set_ylabel('Count')
    ax2.set_title('Distribution of Fourier Scores Across Layers')
    ax2.axvline(0.2, color='orange', linestyle='--', label='Threshold')
    ax2.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to: {save_path}")
    else:
        plt.show()

    return fig


# ============================================================================
# Main Analysis Pipeline
# ============================================================================

def run_detailed_analysis(model_path, M=None, n_heads=4, save_dir=None):
    """
    Run comprehensive detailed Fourier analysis.
    """
    print("=" * 60)
    print("Detailed Transformer Fourier Analysis")
    print("=" * 60)

    # Load model
    if os.path.isdir(model_path):
        checkpoints = sorted(glob.glob(os.path.join(model_path, "model*.pt")))
        model_file = checkpoints[-1]
    else:
        model_file = model_path

    print(f"\nLoading: {model_file}")
    data = torch.load(model_file, map_location='cpu')
    state_dict = data['model']

    # Infer M
    if M is None:
        if 'embedding.weight' in state_dict:
            M = state_dict['embedding.weight'].shape[0]
        elif 'V.weight' in state_dict:
            M = state_dict['V.weight'].shape[0]

    print(f"Group order M = {M}")

    weights = extract_transformer_weights(state_dict)
    hidden_size = weights['embedding'].shape[1] if weights['embedding'] is not None else None

    print(f"Hidden size = {hidden_size}")
    print(f"Number of heads = {n_heads}")

    # 1. Location analysis
    print("\n" + "-" * 40)
    print("1. Fourier Structure Location Analysis")
    print("-" * 40)

    location_report = locate_fourier_structures(model_file, M=M)

    print(f"\nStrong Fourier locations (top 5):")
    for name, score in location_report['strong_fourier_locations'][:5]:
        print(f"  {name}: {score:.4f}")

    print(f"\nWeak Fourier locations (top 5):")
    for name, score in location_report['weak_fourier_locations'][:5]:
        print(f"  {name}: {score:.4f}")

    # 2. Attention head analysis
    print("\n" + "-" * 40)
    print("2. Per-Attention-Head Analysis")
    print("-" * 40)

    if weights['attention']:
        attn_analysis = analyze_attention_head_fourier(
            weights['attention'], M, hidden_size, n_heads
        )

        for layer_idx, layer_results in attn_analysis.items():
            print(f"\nLayer {layer_idx}:")
            print(f"  Mean QK rank (90%): {layer_results['aggregate'].get('mean_QK_rank', 'N/A'):.1f}")
            print(f"  Mean V rank (90%): {layer_results['aggregate'].get('mean_V_rank', 'N/A'):.1f}")
            if 'O_rank_90' in layer_results['aggregate']:
                print(f"  Output proj rank (90%): {layer_results['aggregate']['O_rank_90']}")

    # 3. FFN analysis
    print("\n" + "-" * 40)
    print("3. FFN Layer Analysis")
    print("-" * 40)

    if weights['ffn']:
        ffn_analysis = analyze_ffn_fourier_detailed(weights['ffn'], M, hidden_size)

        for layer_idx, layer_results in ffn_analysis.items():
            print(f"\nLayer {layer_idx}:")
            print(f"  W1: shape={layer_results['W1']['shape']}, rank_90={layer_results['W1']['rank_90']}")
            print(f"  W2: shape={layer_results['W2']['shape']}, rank_90={layer_results['W2']['rank_90']}")
            print(f"  W_combined: rank_90={layer_results['W_combined']['rank_90']}")
            if 'fourier_diagonality' in layer_results:
                print(f"  Fourier diagonality: {layer_results['fourier_diagonality']:.4f}")

    # 4. Comparison with MLP patterns
    print("\n" + "-" * 40)
    print("4. Comparison with MLP Patterns")
    print("-" * 40)

    mlp_comparison = compare_with_mlp_pattern(weights, M, hidden_size)

    if 'embedding' in mlp_comparison:
        print(f"\nEmbedding layer:")
        print(f"  MLP similarity (sparsity): {mlp_comparison['embedding']['mlp_similarity']:.4f}")
        print(f"  Unique dominant frequencies: {mlp_comparison['embedding']['unique_dominant_freqs']}/{M}")

    if 'output_V' in mlp_comparison:
        print(f"\nOutput V layer:")
        print(f"  MLP similarity (sparsity): {mlp_comparison['output_V']['mlp_similarity']:.4f}")
        print(f"  Unique dominant frequencies: {mlp_comparison['output_V']['unique_dominant_freqs']}/{M}")

    # Save visualizations
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

        # Location report
        plot_fourier_location_report(
            location_report,
            os.path.join(save_dir, 'fourier_locations.pdf')
        )

        # Attention analysis
        if weights['attention']:
            plot_attention_head_analysis(
                attn_analysis,
                os.path.join(save_dir, 'attention_heads.pdf')
            )

    print("\n" + "=" * 60)
    print("Analysis complete!")
    if save_dir:
        print(f"Plots saved to: {save_dir}")
    print("=" * 60)

    return {
        'location_report': location_report,
        'attention_analysis': attn_analysis if weights['attention'] else None,
        'ffn_analysis': ffn_analysis if weights['ffn'] else None,
        'mlp_comparison': mlp_comparison,
    }


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detailed Transformer Fourier Analysis")
    parser.add_argument("model_path", type=str, help="Path to model checkpoint or folder")
    parser.add_argument("--M", type=int, default=None, help="Group order")
    parser.add_argument("--n_heads", type=int, default=4, help="Number of attention heads")
    parser.add_argument("--save_dir", type=str, default=None, help="Directory to save plots")

    args = parser.parse_args()

    results = run_detailed_analysis(
        args.model_path,
        M=args.M,
        n_heads=args.n_heads,
        save_dir=args.save_dir
    )
