"""
Visualization tools for LOB Transformer analysis and monitoring.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
from pathlib import Path
from collections import Counter

from lobgpt.tokenizer import LOBTokenizer
from lobgpt.models.lob_transformer import LOBTransformer


class LOBTransformerVisualizer:
    """Visualization tools for LOB Transformer models."""

    def __init__(self, model: LOBTransformer, tokenizer: LOBTokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def plot_attention_patterns(
        self,
        input_ids: torch.Tensor,
        layer_idx: int = -1,
        head_idx: int = 0,
        save_path: Optional[str] = None
    ):
        """Visualize attention patterns for a sequence."""
        self.model.eval()

        with torch.no_grad():
            # Get attention weights
            logits, attention_weights = self.model(
                input_ids.unsqueeze(0),
                return_attention=True
            )

            if attention_weights is None:
                print("Model not configured to return attention weights")
                return

            # Extract specific layer and head
            layer_attn = attention_weights[layer_idx, 0, head_idx]  # [seq_len, seq_len]
            seq_len = layer_attn.size(0)

            # Create figure
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

            # Attention heatmap
            sns.heatmap(
                layer_attn.cpu().numpy(),
                ax=ax1,
                cmap='Blues',
                cbar=True,
                square=True
            )
            ax1.set_title(f'Attention Weights (Layer {layer_idx}, Head {head_idx})')
            ax1.set_xlabel('Key Position')
            ax1.set_ylabel('Query Position')

            # Average attention per position
            avg_attention = layer_attn.mean(dim=1).cpu().numpy()
            ax2.bar(range(seq_len), avg_attention)
            ax2.set_title('Average Attention per Position')
            ax2.set_xlabel('Position')
            ax2.set_ylabel('Average Attention')

            plt.tight_layout()

            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            else:
                plt.show()

    def plot_token_embeddings(
        self,
        method: str = "pca",
        n_tokens: int = 1000,
        save_path: Optional[str] = None
    ):
        """Visualize token embeddings using dimensionality reduction."""
        from sklearn.decomposition import PCA
        from sklearn.manifold import TSNE

        # Get token embeddings
        embeddings = self.model.token_embedding.weight.data.cpu().numpy()
        vocab_size, embed_dim = embeddings.shape

        # Sample tokens if vocabulary is large
        if vocab_size > n_tokens:
            indices = np.random.choice(vocab_size, n_tokens, replace=False)
            embeddings = embeddings[indices]
        else:
            indices = np.arange(vocab_size)

        # Dimensionality reduction
        if method == "pca":
            reducer = PCA(n_components=2)
            coords = reducer.fit_transform(embeddings)
            title = f"Token Embeddings (PCA) - {len(indices)} tokens"
        elif method == "tsne":
            reducer = TSNE(n_components=2, random_state=42)
            coords = reducer.fit_transform(embeddings)
            title = f"Token Embeddings (t-SNE) - {len(indices)} tokens"
        else:
            raise ValueError(f"Unknown method: {method}")

        # Plot
        plt.figure(figsize=(12, 8))
        scatter = plt.scatter(coords[:, 0], coords[:, 1], alpha=0.6, s=30)

        # Add annotations for some tokens
        n_annotate = min(20, len(indices))
        annotate_indices = np.random.choice(len(indices), n_annotate, replace=False)

        for i in annotate_indices:
            token_id = indices[i]
            try:
                # Try to decode token for annotation
                event = self.tokenizer.decode_token(token_id)
                if event:
                    label = f"{event.event_type.name[:3]}-{event.side.name}"
                else:
                    label = str(token_id)
            except:
                label = str(token_id)

            plt.annotate(label, (coords[i, 0], coords[i, 1]),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=8, alpha=0.7)

        plt.title(title)
        plt.xlabel(f"{method.upper()} Component 1")
        plt.ylabel(f"{method.upper()} Component 2")

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()

    def plot_generation_analysis(
        self,
        prompt_tokens: torch.Tensor,
        generated_tokens: torch.Tensor,
        save_path: Optional[str] = None
    ):
        """Analyze and visualize model generation."""
        prompt_len = len(prompt_tokens)
        total_len = len(generated_tokens)
        generated_only = generated_tokens[prompt_len:]

        # Decode sequences
        try:
            prompt_events = [self.tokenizer.decode_token(t.item()) for t in prompt_tokens]
            generated_events = [self.tokenizer.decode_token(t.item()) for t in generated_only]

            # Filter out None values
            prompt_events = [e for e in prompt_events if e is not None]
            generated_events = [e for e in generated_events if e is not None]

        except Exception as e:
            print(f"Error decoding tokens: {e}")
            return

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Event type distribution
        prompt_types = [e.event_type.value for e in prompt_events]
        gen_types = [e.event_type.value for e in generated_events]

        type_counts_prompt = pd.Series(prompt_types).value_counts()
        type_counts_gen = pd.Series(gen_types).value_counts()

        x = np.arange(len(type_counts_prompt))
        width = 0.35

        axes[0,0].bar(x - width/2, type_counts_prompt.values, width,
                     label='Prompt', alpha=0.7)
        axes[0,0].bar(x + width/2, type_counts_gen.values, width,
                     label='Generated', alpha=0.7)
        axes[0,0].set_xlabel('Event Type')
        axes[0,0].set_ylabel('Count')
        axes[0,0].set_title('Event Type Distribution')
        axes[0,0].set_xticks(x)
        axes[0,0].set_xticklabels(type_counts_prompt.index)
        axes[0,0].legend()

        # Side distribution
        generated_side_counts = Counter(event.side.name for event in generated_events)

        axes[0,1].pie([generated_side_counts.get('BID', 0), generated_side_counts.get('ASK', 0)],
                     labels=['BID', 'ASK'],
                     autopct='%1.1f%%',
                     startangle=90)
        axes[0,1].set_title('Generated Side Distribution')

        # Price bucket distribution
        gen_price_levels = [e.price_bucket.value for e in generated_events]
        price_level_counts = pd.Series(gen_price_levels).value_counts()

        axes[1,0].bar(range(len(price_level_counts)), price_level_counts.values)
        axes[1,0].set_xlabel('Price Bucket')
        axes[1,0].set_ylabel('Count')
        axes[1,0].set_title('Generated Price Bucket Distribution')
        axes[1,0].set_xticks(range(len(price_level_counts)))
        axes[1,0].set_xticklabels(price_level_counts.index, rotation=45)

        # Sequence position analysis
        position_types = {}
        for i, event in enumerate(generated_events):
            pos = i // 10  # Group by position buckets
            if pos not in position_types:
                position_types[pos] = []
            position_types[pos].append(event.event_type.value)

        # Plot event type diversity over sequence
        positions = []
        diversities = []
        for pos, types in position_types.items():
            positions.append(pos)
            diversities.append(len(set(types)))

        axes[1,1].plot(positions, diversities, 'o-')
        axes[1,1].set_xlabel('Sequence Position (x10)')
        axes[1,1].set_ylabel('Event Type Diversity')
        axes[1,1].set_title('Event Diversity Over Sequence')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()

    def plot_training_curves(
        self,
        train_losses: List[float],
        val_losses: List[float],
        train_accs: List[float],
        val_accs: List[float],
        save_path: Optional[str] = None
    ):
        """Plot training curves."""
        epochs = range(1, len(train_losses) + 1)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        # Loss curves
        ax1.plot(epochs, train_losses, 'b-', label='Training Loss')
        ax1.plot(epochs, val_losses, 'r-', label='Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True)

        # Accuracy curves
        ax2.plot(epochs, train_accs, 'b-', label='Training Accuracy')
        ax2.plot(epochs, val_accs, 'r-', label='Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Training and Validation Accuracy')
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()

    def create_interactive_attention_plot(
        self,
        input_ids: torch.Tensor,
        save_path: Optional[str] = None
    ):
        """Create interactive attention visualization using Plotly."""
        self.model.eval()

        with torch.no_grad():
            logits, attention_weights = self.model(
                input_ids.unsqueeze(0),
                return_attention=True
            )

            if attention_weights is None:
                print("Model not configured to return attention weights")
                return

            seq_len = input_ids.size(0)
            n_layers, _, n_heads, _, _ = attention_weights.shape

            # Create subplots for each layer
            fig = make_subplots(
                rows=n_layers,
                cols=1,
                subplot_titles=[f"Layer {i}" for i in range(n_layers)],
                vertical_spacing=0.02
            )

            for layer in range(n_layers):
                # Average across heads
                layer_attn = attention_weights[layer, 0].mean(dim=0)

                # Add heatmap
                fig.add_trace(
                    go.Heatmap(
                        z=layer_attn.cpu().numpy(),
                        colorscale='Blues',
                        showscale=True if layer == 0 else False
                    ),
                    row=layer + 1,
                    col=1
                )

            fig.update_layout(
                height=300 * n_layers,
                title='Attention Patterns Across Layers'
            )

            if save_path:
                fig.write_html(save_path)
            else:
                fig.show()

    def plot_model_complexity_analysis(self, save_path: Optional[str] = None):
        """Analyze model complexity and parameter distribution."""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        # Analyze parameters by component
        param_breakdown = {}
        for name, module in self.model.named_modules():
            if len(list(module.children())) == 0:  # Leaf modules only
                module_params = sum(p.numel() for p in module.parameters())
                if module_params > 0:
                    param_breakdown[name] = module_params

        # Sort by parameter count
        param_breakdown = dict(sorted(param_breakdown.items(), key=lambda x: x[1], reverse=True))

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Parameter breakdown pie chart (top 10)
        top_10 = dict(list(param_breakdown.items())[:10])
        others = sum(list(param_breakdown.values())[10:])
        if others > 0:
            top_10['Others'] = others

        axes[0,0].pie(top_10.values(), labels=top_10.keys(), autopct='%1.1f%%')
        axes[0,0].set_title('Parameter Distribution by Component')

        # Parameter count by layer type
        layer_types = {}
        for name, params in param_breakdown.items():
            layer_type = name.split('.')[-2] if '.' in name else name
            if layer_type not in layer_types:
                layer_types[layer_type] = 0
            layer_types[layer_type] += params

        axes[0,1].bar(range(len(layer_types)), list(layer_types.values()))
        axes[0,1].set_xlabel('Layer Type')
        axes[0,1].set_ylabel('Parameters')
        axes[0,1].set_title('Parameters by Layer Type')
        axes[0,1].set_xticks(range(len(layer_types)))
        axes[0,1].set_xticklabels(layer_types.keys(), rotation=45)

        # Model statistics
        stats_text = f"""
Model Statistics:
- Total Parameters: {total_params:,}
- Trainable Parameters: {trainable_params:,}
- Model Size (MB): {total_params * 4 / 1024 / 1024:.1f}
- Layers: {self.model.config.n_layers}
- Heads: {self.model.config.n_heads}
- Model Dimension: {self.model.config.d_model}
- Vocab Size: {self.model.config.vocab_size}
        """

        axes[1,0].text(0.1, 0.5, stats_text, fontsize=12, verticalalignment='center')
        axes[1,0].set_xlim(0, 1)
        axes[1,0].set_ylim(0, 1)
        axes[1,0].axis('off')
        axes[1,0].set_title('Model Statistics')

        # Memory usage estimation
        # Rough estimates for different batch sizes
        batch_sizes = [1, 4, 8, 16, 32, 64]
        seq_len = self.model.config.max_seq_length
        d_model = self.model.config.d_model

        memory_estimates = []
        for bs in batch_sizes:
            # Rough estimate: parameters + activations + gradients
            param_memory = total_params * 4  # 4 bytes per float32
            activation_memory = bs * seq_len * d_model * 4 * 2  # Forward + backward
            grad_memory = total_params * 4  # Gradients
            total_memory = (param_memory + activation_memory + grad_memory) / 1024 / 1024  # MB
            memory_estimates.append(total_memory)

        axes[1,1].plot(batch_sizes, memory_estimates, 'o-')
        axes[1,1].set_xlabel('Batch Size')
        axes[1,1].set_ylabel('Estimated Memory (MB)')
        axes[1,1].set_title('Memory Usage vs Batch Size')
        axes[1,1].grid(True)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()


def plot_loss_landscape(
    model: LOBTransformer,
    dataloader,
    device: str = "cuda",
    n_samples: int = 100,
    perturbation_scale: float = 0.1,
    save_path: Optional[str] = None
):
    """Plot loss landscape around current model weights."""
    model.eval()

    # Get a batch of data
    batch = next(iter(dataloader))
    input_ids = batch["input_ids"][:n_samples].to(device)

    # Calculate loss at current weights
    with torch.no_grad():
        if input_ids.size(1) <= 1:
            print("Sequence too short for loss landscape")
            return

        inputs = input_ids[:, :-1]
        targets = input_ids[:, 1:]

        current_logits, _ = model(inputs)
        current_loss = torch.nn.functional.cross_entropy(
            current_logits.reshape(-1, current_logits.size(-1)),
            targets.reshape(-1),
            ignore_index=-100
        ).item()

    # Generate random directions
    directions = []
    for param in model.parameters():
        if param.requires_grad:
            direction = torch.randn_like(param)
            direction = direction / direction.norm() * perturbation_scale
            directions.append(direction)

    # Sample points along random directions
    alphas = np.linspace(-2, 2, 50)
    losses = []

    for alpha in alphas:
        # Perturb weights
        with torch.no_grad():
            for param, direction in zip(model.parameters(), directions):
                if param.requires_grad:
                    param.data += alpha * direction

        # Calculate loss
        model.eval()
        with torch.no_grad():
            logits, _ = model(inputs)
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=-100
            ).item()
            losses.append(loss)

        # Restore weights
        with torch.no_grad():
            for param, direction in zip(model.parameters(), directions):
                if param.requires_grad:
                    param.data -= alpha * direction

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(alphas, losses, 'b-', linewidth=2)
    plt.axvline(x=0, color='r', linestyle='--', alpha=0.7, label='Current Position')
    plt.axhline(y=current_loss, color='r', linestyle='--', alpha=0.7)
    plt.xlabel('Perturbation Scale')
    plt.ylabel('Loss')
    plt.title('Loss Landscape Along Random Direction')
    plt.grid(True, alpha=0.3)
    plt.legend()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
