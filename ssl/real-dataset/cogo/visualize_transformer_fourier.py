"""
Visualization code to analyze Fourier weight structures in Transformer models
and compare with MLP Fourier weight patterns.

This script analyzes:
1. Embedding layer Fourier structure
2. Attention layer (Q, K, V projections) Fourier structure
3. FFN layer Fourier structure
4. Output layer V Fourier structure

Usage:
    python visualize_transformer_fourier.py <model_path_or_folder> [--model_index N] [--compare_mlp <mlp_folder>]
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

# ============================================================================
# Utility Functions
# ============================================================================

def get_fourier_basis(d):
    """Construct Fourier basis matrix for dimension d."""
    return construct_bases(d)


def project_to_fourier(weight, fourier_basis, axis='output'):
    """
    Project weight matrix to Fourier domain.

    Args:
        weight: Weight matrix [out_dim, in_dim] or [in_dim, out_dim]
        fourier_basis: Fourier basis matrix [d, d]
        axis: 'output' or 'input' - which dimension to project

    Returns:
        Fourier-transformed weight matrix
    """
    d = fourier_basis.shape[0]
    weight_c = weight.cfloat()

    if axis == 'output':
        # Project output dimension
        if weight_c.shape[0] == d:
            return weight_c @ fourier_basis.conj() / d
        else:
            return weight_c
    else:
        # Project input dimension
        if weight_c.shape[1] == d:
            return fourier_basis.conj().t() @ weight_c / d
        else:
            return weight_c


def compute_fourier_spectrum(weight, fourier_basis):
    """
    Compute the Fourier spectrum (magnitude) of weight matrix.

    Returns spectrum for each frequency.
    """
    d = fourier_basis.shape[0]
    weight_c = weight.cfloat()

    # Try both dimensions
    spectra = {}

    if weight_c.shape[0] == d:
        proj = weight_c @ fourier_basis.conj() / d
        spectra['output'] = proj.abs().mean(dim=1)  # Average over hidden dim

    if weight_c.shape[1] == d:
        proj = fourier_basis.conj().t() @ weight_c / d
        spectra['input'] = proj.abs().mean(dim=1)

    return spectra


def get_dominant_frequencies(spectrum, threshold=0.1):
    """Get indices of dominant frequencies in spectrum."""
    max_val = spectrum.max()
    return (spectrum > threshold * max_val).nonzero().squeeze(-1).tolist()


# ============================================================================
# Transformer Weight Extraction
# ============================================================================

def extract_transformer_weights(state_dict):
    """
    Extract all relevant weights from a transformer model state dict.

    Returns a dict with:
        - embedding: Embedding weight matrix
        - pos_encoder: Positional encoding
        - attention: Dict with Q, K, V, out projections for each layer
        - ffn: Dict with linear1, linear2 for each layer
        - output_V: Final output projection
    """
    weights = {
        'embedding': None,
        'pos_encoder': None,
        'attention': defaultdict(dict),
        'ffn': defaultdict(dict),
        'output_V': None,
    }

    for key, value in state_dict.items():
        if 'embedding.weight' in key:
            weights['embedding'] = value
        elif 'pos_encoder' in key:
            weights['pos_encoder'] = value
        elif 'V.weight' in key and 'transformer' not in key:
            weights['output_V'] = value
        elif 'transformer_encoder' in key:
            # Parse layer index
            parts = key.split('.')
            # Find layer index - looking for patterns like 'layers.0' or 'layers.1'
            layer_idx = None
            for i, p in enumerate(parts):
                if p == 'layers' and i + 1 < len(parts):
                    try:
                        layer_idx = int(parts[i + 1])
                        break
                    except ValueError:
                        continue

            if layer_idx is not None:
                if 'self_attn' in key:
                    if 'in_proj_weight' in key:
                        # Combined Q, K, V projection
                        weights['attention'][layer_idx]['in_proj'] = value
                    elif 'out_proj.weight' in key:
                        weights['attention'][layer_idx]['out_proj'] = value
                    elif 'q_proj' in key:
                        weights['attention'][layer_idx]['q_proj'] = value
                    elif 'k_proj' in key:
                        weights['attention'][layer_idx]['k_proj'] = value
                    elif 'v_proj' in key:
                        weights['attention'][layer_idx]['v_proj'] = value
                elif 'linear1.weight' in key:
                    weights['ffn'][layer_idx]['linear1'] = value
                elif 'linear2.weight' in key:
                    weights['ffn'][layer_idx]['linear2'] = value

    return weights


def extract_mlp_weights(state_dict):
    """
    Extract weights from MLP model state dict.

    Returns a dict with:
        - W: Hidden layer weight (or Ws for inner product act)
        - V: Output layer weight
    """
    weights = {
        'W': None,
        'Ws': [],
        'V': None,
        'other_layers': [],
    }

    for key, value in state_dict.items():
        if key == 'W.weight':
            weights['W'] = value
        elif key == 'V.weight':
            weights['V'] = value
        elif key.startswith('Ws.'):
            idx = int(key.split('.')[1])
            while len(weights['Ws']) <= idx:
                weights['Ws'].append(None)
            weights['Ws'][idx] = value
        elif 'other_layers' in key and 'weight' in key:
            weights['other_layers'].append(value)

    return weights


# ============================================================================
# Fourier Analysis Functions
# ============================================================================

def analyze_embedding_fourier(embedding_weight, M):
    """
    Analyze Fourier structure in embedding layer.

    Args:
        embedding_weight: [M, hidden_size] embedding matrix
        M: vocabulary size (group order)

    Returns:
        Analysis dict with Fourier projections and statistics
    """
    fourier_basis = get_fourier_basis(M)

    # Project embedding to Fourier domain: [M, hidden] -> [M, hidden]
    # Each column of embedding corresponds to one vocab element
    # Project each hidden dimension's embedding vector
    emb_fourier = fourier_basis.conj().t() @ embedding_weight.cfloat() / M

    # Compute spectrum per frequency
    spectrum = emb_fourier.abs().mean(dim=1)  # Average over hidden dimensions

    # Find dominant frequencies
    dominant_freqs = get_dominant_frequencies(spectrum)

    return {
        'fourier_proj': emb_fourier,
        'spectrum': spectrum,
        'dominant_freqs': dominant_freqs,
        'max_spectrum': spectrum.max().item(),
    }


def analyze_attention_fourier(attn_weights, M, hidden_size):
    """
    Analyze Fourier structure in attention layer weights.
    """
    fourier_basis = get_fourier_basis(M)
    results = {}

    for layer_idx, layer_weights in attn_weights.items():
        layer_results = {}

        for weight_name, weight in layer_weights.items():
            if weight is None:
                continue

            # For attention weights, the relevant dimension is hidden_size
            # which may not equal M, so we analyze the weight spectrum differently
            weight_c = weight.cfloat()

            # Compute singular values to understand weight structure
            U, S, Vh = torch.linalg.svd(weight_c, full_matrices=False)

            layer_results[weight_name] = {
                'shape': weight.shape,
                'singular_values': S,
                'top_singular': S[:10].tolist() if len(S) >= 10 else S.tolist(),
                'rank_90': (S.cumsum(0) / S.sum() < 0.9).sum().item() + 1,
            }

            # If weight dimension matches M, do Fourier analysis
            if weight.shape[0] == M or weight.shape[1] == M:
                spectrum = compute_fourier_spectrum(weight, fourier_basis)
                layer_results[weight_name]['fourier_spectrum'] = spectrum

        results[layer_idx] = layer_results

    return results


def analyze_ffn_fourier(ffn_weights, M, hidden_size):
    """
    Analyze Fourier structure in FFN layer weights.

    FFN structure: x -> linear1 -> activation -> linear2
    linear1: [hidden_size, dim_ffn]
    linear2: [dim_ffn, hidden_size]
    """
    fourier_basis = get_fourier_basis(M)
    results = {}

    for layer_idx, layer_weights in ffn_weights.items():
        layer_results = {}

        for weight_name, weight in layer_weights.items():
            if weight is None:
                continue

            weight_c = weight.cfloat()

            # Compute SVD for structural analysis
            U, S, Vh = torch.linalg.svd(weight_c, full_matrices=False)

            layer_results[weight_name] = {
                'shape': weight.shape,
                'singular_values': S,
                'top_singular': S[:10].tolist() if len(S) >= 10 else S.tolist(),
                'rank_90': (S.cumsum(0) / S.sum() < 0.9).sum().item() + 1,
                'frobenius_norm': weight.norm().item(),
            }

            # If dimension matches M, do Fourier analysis
            if weight.shape[0] == M or weight.shape[1] == M:
                spectrum = compute_fourier_spectrum(weight, fourier_basis)
                layer_results[weight_name]['fourier_spectrum'] = spectrum

        results[layer_idx] = layer_results

    return results


def analyze_output_fourier(V_weight, M):
    """
    Analyze Fourier structure in output layer V.

    V: [M, hidden_size] - maps hidden to output logits
    """
    fourier_basis = get_fourier_basis(M)

    V_c = V_weight.cfloat()

    # Project output dimension to Fourier domain
    # V shape is [M, hidden_size], we want to project the M dimension
    # fourier_basis.conj().t() @ V gives [M, hidden_size] in Fourier domain
    V_fourier = fourier_basis.conj().t() @ V_c / M

    # Spectrum per frequency (average over hidden dim)
    spectrum = V_fourier.abs().mean(dim=1)

    # SVD analysis
    U, S, Vh = torch.linalg.svd(V_c, full_matrices=False)

    return {
        'fourier_proj': V_fourier,
        'spectrum': spectrum,
        'dominant_freqs': get_dominant_frequencies(spectrum),
        'singular_values': S,
        'rank_90': (S.cumsum(0) / S.sum() < 0.9).sum().item() + 1,
    }


def analyze_mlp_fourier(mlp_weights, M):
    """
    Analyze Fourier structure in MLP model weights.
    Similar to the existing analysis but unified interface.
    """
    fourier_basis = get_fourier_basis(M)
    results = {}

    # Analyze W layer
    if mlp_weights['W'] is not None:
        W = mlp_weights['W'].cfloat()
        # W shape: [hidden_size, num_ops * M]
        # Split by input dimension
        num_ops = W.shape[1] // M

        W_fourier_list = []
        for op_idx in range(num_ops):
            W_op = W[:, op_idx * M : (op_idx + 1) * M]
            W_fourier = W_op @ fourier_basis.conj() / M
            W_fourier_list.append(W_fourier)

        results['W'] = {
            'fourier_proj': W_fourier_list,
            'spectra': [wf.abs().mean(dim=0) for wf in W_fourier_list],
        }

    elif mlp_weights['Ws']:
        results['Ws'] = []
        for i, W in enumerate(mlp_weights['Ws']):
            if W is None:
                continue
            W_c = W.cfloat()
            W_fourier = W_c @ fourier_basis.conj() / M
            results['Ws'].append({
                'fourier_proj': W_fourier,
                'spectrum': W_fourier.abs().mean(dim=0),
            })

    # Analyze V layer
    if mlp_weights['V'] is not None:
        V = mlp_weights['V'].cfloat()
        # V shape is [M, hidden_size], project M dimension to Fourier
        V_fourier = fourier_basis.conj().t() @ V / M
        results['V'] = {
            'fourier_proj': V_fourier,
            'spectrum': V_fourier.abs().mean(dim=1),
        }

    return results


# ============================================================================
# Visualization Functions
# ============================================================================

def plot_fourier_spectrum(spectrum, title, ax=None, log_scale=False):
    """Plot Fourier spectrum as bar chart."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))

    x = torch.arange(len(spectrum))
    values = spectrum.cpu().numpy() if isinstance(spectrum, torch.Tensor) else spectrum

    if log_scale:
        values = np.log10(values + 1e-10)

    ax.bar(x, values, alpha=0.7)
    ax.set_xlabel('Frequency index k')
    ax.set_ylabel('Magnitude' + (' (log10)' if log_scale else ''))
    ax.set_title(title)

    return ax


def plot_fourier_heatmap(fourier_proj, title, ax=None, vmax=None):
    """Plot Fourier projection as heatmap."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))

    data = fourier_proj.abs().cpu().numpy()

    im = ax.imshow(data, aspect='auto', cmap='viridis', vmax=vmax)
    ax.set_xlabel('Hidden dimension')
    ax.set_ylabel('Frequency index k')
    ax.set_title(title)
    plt.colorbar(im, ax=ax)

    return ax


def plot_singular_values(S, title, ax=None, top_k=50):
    """Plot singular values."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))

    S_np = S[:top_k].cpu().numpy() if isinstance(S, torch.Tensor) else S[:top_k]

    ax.semilogy(S_np, 'o-')
    ax.set_xlabel('Singular value index')
    ax.set_ylabel('Singular value (log scale)')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    return ax


def visualize_transformer_analysis(model_path, M=None, save_dir=None):
    """
    Main visualization function for transformer model.

    Args:
        model_path: Path to model checkpoint or folder containing checkpoints
        M: Group order (if None, inferred from model)
        save_dir: Directory to save plots (if None, show interactively)
    """
    # Load model
    if os.path.isdir(model_path):
        checkpoints = sorted(glob.glob(os.path.join(model_path, "model*.pt")))
        if not checkpoints:
            raise ValueError(f"No model checkpoints found in {model_path}")
        model_file = checkpoints[-1]  # Use latest
    else:
        model_file = model_path

    print(f"Loading model from: {model_file}")
    data = torch.load(model_file, map_location='cpu')
    state_dict = data['model']

    # Infer M from embedding or V weight
    if M is None:
        if 'embedding.weight' in state_dict:
            M = state_dict['embedding.weight'].shape[0]
        elif 'V.weight' in state_dict:
            M = state_dict['V.weight'].shape[0]
        else:
            raise ValueError("Cannot infer M from model, please specify")

    print(f"Group order M = {M}")

    # Check if this is a transformer or MLP
    is_transformer = 'embedding.weight' in state_dict and 'transformer_encoder' in str(state_dict.keys())

    if is_transformer:
        weights = extract_transformer_weights(state_dict)
        hidden_size = weights['embedding'].shape[1] if weights['embedding'] is not None else None

        print(f"\n=== Transformer Model Analysis ===")
        print(f"Hidden size: {hidden_size}")

        # Create figure for comprehensive visualization
        fig = plt.figure(figsize=(20, 16))

        # 1. Embedding analysis
        if weights['embedding'] is not None:
            print("\n--- Embedding Layer ---")
            emb_analysis = analyze_embedding_fourier(weights['embedding'], M)
            print(f"Dominant frequencies: {emb_analysis['dominant_freqs'][:10]}")
            print(f"Max spectrum value: {emb_analysis['max_spectrum']:.4f}")

            ax1 = fig.add_subplot(3, 3, 1)
            plot_fourier_spectrum(emb_analysis['spectrum'], 'Embedding Fourier Spectrum', ax1)

            ax2 = fig.add_subplot(3, 3, 2)
            plot_fourier_heatmap(emb_analysis['fourier_proj'], 'Embedding Fourier Projection', ax2)

        # 2. Output V analysis
        if weights['output_V'] is not None:
            print("\n--- Output Layer V ---")
            V_analysis = analyze_output_fourier(weights['output_V'], M)
            print(f"Dominant frequencies: {V_analysis['dominant_freqs'][:10]}")
            print(f"Effective rank (90%): {V_analysis['rank_90']}")

            ax3 = fig.add_subplot(3, 3, 3)
            plot_fourier_spectrum(V_analysis['spectrum'], 'Output V Fourier Spectrum', ax3)

            ax4 = fig.add_subplot(3, 3, 4)
            plot_fourier_heatmap(V_analysis['fourier_proj'], 'Output V Fourier Projection', ax4)

            ax5 = fig.add_subplot(3, 3, 5)
            plot_singular_values(V_analysis['singular_values'], 'Output V Singular Values', ax5)

        # 3. Attention analysis
        if weights['attention']:
            print("\n--- Attention Layers ---")
            attn_analysis = analyze_attention_fourier(weights['attention'], M, hidden_size)
            for layer_idx, layer_results in attn_analysis.items():
                print(f"  Layer {layer_idx}:")
                for weight_name, results in layer_results.items():
                    print(f"    {weight_name}: shape={results['shape']}, rank_90={results['rank_90']}")

        # 4. FFN analysis
        if weights['ffn']:
            print("\n--- FFN Layers ---")
            ffn_analysis = analyze_ffn_fourier(weights['ffn'], M, hidden_size)
            for layer_idx, layer_results in ffn_analysis.items():
                print(f"  Layer {layer_idx}:")
                for weight_name, results in layer_results.items():
                    print(f"    {weight_name}: shape={results['shape']}, rank_90={results['rank_90']}")
                    if 'fourier_spectrum' in results:
                        for dim, spec in results['fourier_spectrum'].items():
                            print(f"      Fourier spectrum ({dim}): max={spec.max():.4f}")

            # Plot FFN analysis
            ax6 = fig.add_subplot(3, 3, 6)
            if 0 in ffn_analysis and 'linear1' in ffn_analysis[0]:
                S = ffn_analysis[0]['linear1']['singular_values']
                plot_singular_values(S, 'FFN Linear1 (Layer 0) Singular Values', ax6)

            ax7 = fig.add_subplot(3, 3, 7)
            if 0 in ffn_analysis and 'linear2' in ffn_analysis[0]:
                S = ffn_analysis[0]['linear2']['singular_values']
                plot_singular_values(S, 'FFN Linear2 (Layer 0) Singular Values', ax7)

    else:
        # MLP model analysis
        weights = extract_mlp_weights(state_dict)
        print(f"\n=== MLP Model Analysis ===")

        fig = plt.figure(figsize=(16, 12))

        mlp_analysis = analyze_mlp_fourier(weights, M)

        if 'W' in mlp_analysis:
            print("\n--- W Layer ---")
            for i, spec in enumerate(mlp_analysis['W']['spectra']):
                print(f"  Op {i}: max spectrum = {spec.max():.4f}")

            ax1 = fig.add_subplot(2, 2, 1)
            plot_fourier_spectrum(mlp_analysis['W']['spectra'][0], 'W (Op 0) Fourier Spectrum', ax1)

            ax2 = fig.add_subplot(2, 2, 2)
            plot_fourier_heatmap(mlp_analysis['W']['fourier_proj'][0].t(), 'W (Op 0) Fourier Projection', ax2)

        if 'V' in mlp_analysis:
            print("\n--- V Layer ---")
            print(f"  Max spectrum = {mlp_analysis['V']['spectrum'].max():.4f}")

            ax3 = fig.add_subplot(2, 2, 3)
            plot_fourier_spectrum(mlp_analysis['V']['spectrum'], 'V Fourier Spectrum', ax3)

            ax4 = fig.add_subplot(2, 2, 4)
            plot_fourier_heatmap(mlp_analysis['V']['fourier_proj'], 'V Fourier Projection', ax4)

    plt.tight_layout()

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, 'fourier_analysis.pdf')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nSaved to: {save_path}")
    else:
        plt.show()

    return fig


def compare_transformer_mlp_fourier(transformer_path, mlp_path, M=None, save_dir=None):
    """
    Compare Fourier weight structures between transformer and MLP models.
    """
    print("=== Comparing Transformer vs MLP Fourier Structures ===\n")

    # Load both models
    t_data = torch.load(transformer_path, map_location='cpu')
    m_data = torch.load(mlp_path, map_location='cpu')

    t_state = t_data['model']
    m_state = m_data['model']

    # Infer M
    if M is None:
        if 'embedding.weight' in t_state:
            M = t_state['embedding.weight'].shape[0]
        elif 'V.weight' in t_state:
            M = t_state['V.weight'].shape[0]

    print(f"Group order M = {M}")

    t_weights = extract_transformer_weights(t_state)
    m_weights = extract_mlp_weights(m_state)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Transformer embedding vs MLP W
    if t_weights['embedding'] is not None:
        t_emb_analysis = analyze_embedding_fourier(t_weights['embedding'], M)
        plot_fourier_spectrum(t_emb_analysis['spectrum'], 'Transformer Embedding', axes[0, 0])

    if m_weights['W'] is not None:
        m_analysis = analyze_mlp_fourier(m_weights, M)
        if 'W' in m_analysis:
            plot_fourier_spectrum(m_analysis['W']['spectra'][0], 'MLP W (Op 0)', axes[1, 0])

    # Output V comparison
    if t_weights['output_V'] is not None:
        t_V_analysis = analyze_output_fourier(t_weights['output_V'], M)
        plot_fourier_spectrum(t_V_analysis['spectrum'], 'Transformer Output V', axes[0, 1])
        plot_singular_values(t_V_analysis['singular_values'], 'Transformer V Singular Values', axes[0, 2])

    if m_weights['V'] is not None:
        m_analysis = analyze_mlp_fourier(m_weights, M)
        if 'V' in m_analysis:
            plot_fourier_spectrum(m_analysis['V']['spectrum'], 'MLP V', axes[1, 1])

        # Compute MLP V singular values
        V_c = m_weights['V'].cfloat()
        U, S, Vh = torch.linalg.svd(V_c, full_matrices=False)
        plot_singular_values(S, 'MLP V Singular Values', axes[1, 2])

    plt.tight_layout()

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, 'transformer_mlp_comparison.pdf')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nSaved to: {save_path}")
    else:
        plt.show()

    return fig


def visualize_fourier_evolution(model_folder, indices=None, M=None, save_dir=None):
    """
    Visualize how Fourier structure evolves during training.

    Args:
        model_folder: Folder containing model checkpoints
        indices: List of epoch indices to analyze (if None, sample automatically)
        M: Group order
        save_dir: Directory to save plots
    """
    checkpoints = sorted(glob.glob(os.path.join(model_folder, "model*.pt")))
    if not checkpoints:
        raise ValueError(f"No checkpoints found in {model_folder}")

    print(f"Found {len(checkpoints)} checkpoints")

    # Sample checkpoints if indices not specified
    if indices is None:
        n_samples = min(10, len(checkpoints))
        step = len(checkpoints) // n_samples
        indices = list(range(0, len(checkpoints), step))[:n_samples]

    # Load first checkpoint to determine model type and M
    first_data = torch.load(checkpoints[0], map_location='cpu')
    first_state = first_data['model']

    is_transformer = 'embedding.weight' in first_state and 'transformer_encoder' in str(first_state.keys())

    if M is None:
        if 'embedding.weight' in first_state:
            M = first_state['embedding.weight'].shape[0]
        elif 'V.weight' in first_state:
            M = first_state['V.weight'].shape[0]

    print(f"Model type: {'Transformer' if is_transformer else 'MLP'}")
    print(f"Group order M = {M}")

    # Collect spectra over time
    epochs = []
    embedding_spectra = []
    V_spectra = []

    for idx in indices:
        if idx >= len(checkpoints):
            continue

        data = torch.load(checkpoints[idx], map_location='cpu')
        state = data['model']
        epoch = data['results'][-1]['epoch'] if 'results' in data else idx
        epochs.append(epoch)

        if is_transformer:
            weights = extract_transformer_weights(state)

            if weights['embedding'] is not None:
                emb_analysis = analyze_embedding_fourier(weights['embedding'], M)
                embedding_spectra.append(emb_analysis['spectrum'])

            if weights['output_V'] is not None:
                V_analysis = analyze_output_fourier(weights['output_V'], M)
                V_spectra.append(V_analysis['spectrum'])
        else:
            weights = extract_mlp_weights(state)
            mlp_analysis = analyze_mlp_fourier(weights, M)

            if 'W' in mlp_analysis:
                embedding_spectra.append(mlp_analysis['W']['spectra'][0])

            if 'V' in mlp_analysis:
                V_spectra.append(mlp_analysis['V']['spectrum'])

    # Plot evolution
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    if embedding_spectra:
        embedding_spectra = torch.stack(embedding_spectra).cpu().numpy()
        im1 = axes[0].imshow(embedding_spectra.T, aspect='auto', cmap='viridis',
                             extent=[epochs[0], epochs[-1], M-1, 0])
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Frequency index k')
        axes[0].set_title('Embedding/W Fourier Spectrum Evolution')
        plt.colorbar(im1, ax=axes[0])

    if V_spectra:
        V_spectra = torch.stack(V_spectra).cpu().numpy()
        im2 = axes[1].imshow(V_spectra.T, aspect='auto', cmap='viridis',
                             extent=[epochs[0], epochs[-1], M-1, 0])
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Frequency index k')
        axes[1].set_title('Output V Fourier Spectrum Evolution')
        plt.colorbar(im2, ax=axes[1])

    plt.tight_layout()

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, 'fourier_evolution.pdf')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nSaved to: {save_path}")
    else:
        plt.show()

    return fig


def analyze_frequency_localization(model_path, M=None, threshold=0.1):
    """
    Analyze how localized the Fourier weights are to specific frequencies.

    Returns metrics about frequency sparsity and localization.
    """
    data = torch.load(model_path, map_location='cpu')
    state = data['model']

    if M is None:
        if 'embedding.weight' in state:
            M = state['embedding.weight'].shape[0]
        elif 'V.weight' in state:
            M = state['V.weight'].shape[0]

    is_transformer = 'embedding.weight' in state and 'transformer_encoder' in str(state.keys())

    results = {
        'M': M,
        'model_type': 'transformer' if is_transformer else 'mlp',
        'layers': {},
    }

    fourier_basis = get_fourier_basis(M)

    if is_transformer:
        weights = extract_transformer_weights(state)

        if weights['embedding'] is not None:
            emb_analysis = analyze_embedding_fourier(weights['embedding'], M)
            spec = emb_analysis['spectrum']
            spec_norm = spec / spec.max()

            results['layers']['embedding'] = {
                'dominant_freqs': emb_analysis['dominant_freqs'],
                'sparsity': (spec_norm > threshold).sum().item() / M,
                'entropy': -(spec_norm * torch.log(spec_norm + 1e-10)).sum().item(),
            }

        if weights['output_V'] is not None:
            V_analysis = analyze_output_fourier(weights['output_V'], M)
            spec = V_analysis['spectrum']
            spec_norm = spec / spec.max()

            results['layers']['output_V'] = {
                'dominant_freqs': V_analysis['dominant_freqs'],
                'sparsity': (spec_norm > threshold).sum().item() / M,
                'entropy': -(spec_norm * torch.log(spec_norm + 1e-10)).sum().item(),
            }
    else:
        weights = extract_mlp_weights(state)
        mlp_analysis = analyze_mlp_fourier(weights, M)

        if 'W' in mlp_analysis:
            spec = mlp_analysis['W']['spectra'][0]
            spec_norm = spec / spec.max()

            results['layers']['W'] = {
                'sparsity': (spec_norm > threshold).sum().item() / M,
                'entropy': -(spec_norm * torch.log(spec_norm + 1e-10)).sum().item(),
            }

        if 'V' in mlp_analysis:
            spec = mlp_analysis['V']['spectrum']
            spec_norm = spec / spec.max()

            results['layers']['V'] = {
                'sparsity': (spec_norm > threshold).sum().item() / M,
                'entropy': -(spec_norm * torch.log(spec_norm + 1e-10)).sum().item(),
            }

    return results


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize Fourier weight structures in Transformer/MLP models")
    parser.add_argument("model_path", type=str, help="Path to model checkpoint or folder")
    parser.add_argument("--M", type=int, default=None, help="Group order (inferred if not specified)")
    parser.add_argument("--save_dir", type=str, default=None, help="Directory to save plots")
    parser.add_argument("--compare_mlp", type=str, default=None, help="MLP model path for comparison")
    parser.add_argument("--evolution", action="store_true", help="Plot Fourier evolution over training")
    parser.add_argument("--localization", action="store_true", help="Analyze frequency localization")

    args = parser.parse_args()

    if args.localization:
        results = analyze_frequency_localization(args.model_path, M=args.M)
        print("\n=== Frequency Localization Analysis ===")
        print(f"Model type: {results['model_type']}")
        print(f"Group order M = {results['M']}")
        for layer_name, layer_results in results['layers'].items():
            print(f"\n{layer_name}:")
            for k, v in layer_results.items():
                if isinstance(v, list):
                    print(f"  {k}: {v[:10]}{'...' if len(v) > 10 else ''}")
                else:
                    print(f"  {k}: {v:.4f}")
    elif args.evolution:
        visualize_fourier_evolution(args.model_path, M=args.M, save_dir=args.save_dir)
    elif args.compare_mlp:
        compare_transformer_mlp_fourier(args.model_path, args.compare_mlp, M=args.M, save_dir=args.save_dir)
    else:
        visualize_transformer_analysis(args.model_path, M=args.M, save_dir=args.save_dir)
