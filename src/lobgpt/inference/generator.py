"""
Advanced inference and generation utilities for LOB Transformer.
"""

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import torch
import torch.nn.functional as F

from lobgpt.models.lob_transformer import LOBTransformer
from lobgpt.tokenizer import EventType, LOBTokenizer, TokenComponents


@dataclass
class GenerationConfig:
    """Configuration for model generation."""

    # Basic generation
    max_length: int = 100
    temperature: float = 1.0
    do_sample: bool = True

    # Sampling strategies
    top_k: Optional[int] = 50
    top_p: Optional[float] = None
    typical_p: Optional[float] = None

    # Repetition control
    repetition_penalty: float = 1.0
    no_repeat_ngram_size: int = 0

    # Stopping criteria
    eos_token_id: Optional[int] = None
    pad_token_id: Optional[int] = 0

    # Financial constraints
    enforce_event_rules: bool = True
    min_price_change: float = 0.0
    max_price_change: float = 1.0

    # Output control
    return_probabilities: bool = False
    return_attention: bool = False


class ConstrainedSampler:
    """Sampler that enforces financial market rules."""

    def __init__(self, tokenizer: LOBTokenizer, config: GenerationConfig):
        self.tokenizer = tokenizer
        self.config = config

    def apply_constraints(
        self,
        logits: torch.Tensor,
        generated_tokens: torch.Tensor,
        current_step: int
    ) -> torch.Tensor:
        """Apply financial constraints to logits."""
        if not self.config.enforce_event_rules:
            return logits

        # Get last few tokens for context
        if generated_tokens.size(1) > 0:
            last_token = generated_tokens[0, -1].item()
            try:
                last_event = self.tokenizer.decode_token(last_token)
            except:
                last_event = None
        else:
            last_event = None

        # Apply constraints based on last event
        if last_event is not None:
            logits = self._apply_event_sequence_rules(logits, last_event)
            logits = self._apply_price_constraints(logits, generated_tokens)

        return logits

    def _apply_event_sequence_rules(
        self,
        logits: torch.Tensor,
        last_event: TokenComponents
    ) -> torch.Tensor:
        """Apply rules about valid event sequences."""
        squeezed = False
        if logits.ndim == 1:
            logits = logits.unsqueeze(0)
            squeezed = True

        if last_event.event_type == EventType.ADD:
            cancel_ids = [
                token_id
                for token_id, token_obj in self.tokenizer.id_to_token.items()
                if token_obj.event_type == EventType.CANCEL
            ]
            if cancel_ids:
                logits[..., cancel_ids] *= 0.5

        return logits.squeeze(0) if squeezed else logits

    def _apply_price_constraints(
        self,
        logits: torch.Tensor,
        generated_tokens: torch.Tensor
    ) -> torch.Tensor:
        """Apply constraints on price movements."""
        # This is a simplified example - in practice, you'd need
        # to track actual price levels and enforce realistic movements
        return logits


class LOBGenerator:
    """Advanced generator for LOB event sequences."""

    def __init__(self, model: LOBTransformer, tokenizer: LOBTokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.model.eval()

    @torch.no_grad()
    def generate(
        self,
        prompt_tokens: torch.Tensor,
        config: GenerationConfig
    ) -> Dict[str, torch.Tensor]:
        """Generate LOB sequences with advanced sampling."""
        batch_size = prompt_tokens.size(0)
        device = prompt_tokens.device

        # Initialize generation
        generated = prompt_tokens.clone()
        sampler = ConstrainedSampler(self.tokenizer, config)

        # Storage for additional outputs
        all_probs = [] if config.return_probabilities else None
        all_attention = [] if config.return_attention else None

        for step in range(config.max_length):
            # Get logits
            logits, attention = self.model(
                generated,
                return_attention=config.return_attention
            )

            next_token_logits = logits[:, -1, :]

            # Apply temperature
            next_token_logits = next_token_logits / config.temperature

            # Apply constraints
            next_token_logits = sampler.apply_constraints(
                next_token_logits, generated, step
            )

            # Apply repetition penalty
            if config.repetition_penalty != 1.0:
                next_token_logits = self._apply_repetition_penalty(
                    next_token_logits, generated, config.repetition_penalty
                )

            # Apply sampling strategies
            if config.do_sample:
                next_token = self._sample_token(next_token_logits, config)
            else:
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

            # Store probabilities if requested
            if config.return_probabilities:
                probs = F.softmax(next_token_logits, dim=-1)
                all_probs.append(probs)

            # Store attention if requested
            if config.return_attention and attention is not None:
                all_attention.append(attention)

            # Append to sequence
            generated = torch.cat([generated, next_token], dim=-1)

            # Check stopping criteria
            if config.eos_token_id is not None:
                if (next_token == config.eos_token_id).all():
                    break

            # Truncate if exceeding max sequence length
            if generated.size(1) > self.model.config.max_seq_length:
                generated = generated[:, -self.model.config.max_seq_length:]

        # Prepare output
        output = {"sequences": generated}

        if all_probs:
            output["probabilities"] = torch.stack(all_probs, dim=1)

        if all_attention:
            output["attention_weights"] = torch.stack(all_attention, dim=1)

        return output

    def _apply_repetition_penalty(
        self,
        logits: torch.Tensor,
        generated: torch.Tensor,
        penalty: float
    ) -> torch.Tensor:
        """Apply repetition penalty to logits."""
        batch_size, vocab_size = logits.shape

        for i in range(batch_size):
            for token_id in generated[i]:
                if logits[i, token_id] < 0:
                    logits[i, token_id] *= penalty
                else:
                    logits[i, token_id] /= penalty

        return logits

    def _sample_token(
        self,
        logits: torch.Tensor,
        config: GenerationConfig
    ) -> torch.Tensor:
        """Sample next token using specified strategy."""
        # Apply top-k filtering
        if config.top_k is not None and config.top_k > 0:
            top_k = min(config.top_k, logits.size(-1))
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = -float('inf')

        # Apply top-p (nucleus) filtering
        if config.top_p is not None and config.top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            # Remove tokens with cumulative probability above threshold
            sorted_indices_to_remove = cumulative_probs > config.top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            indices_to_remove = sorted_indices_to_remove.scatter(
                1, sorted_indices, sorted_indices_to_remove
            )
            logits[indices_to_remove] = -float('inf')

        # Apply typical-p filtering
        if config.typical_p is not None:
            logits = self._apply_typical_p_filtering(logits, config.typical_p)

        # Sample
        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1)

    def _apply_typical_p_filtering(
        self,
        logits: torch.Tensor,
        typical_p: float
    ) -> torch.Tensor:
        """Apply typical-p filtering (Meister et al., 2022)."""
        probs = F.softmax(logits, dim=-1)

        # Calculate entropy
        entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1, keepdim=True)

        # Calculate conditional information
        info = -torch.log(probs + 1e-10)

        # Typical set
        typical_mask = torch.abs(info - entropy) < typical_p

        # Zero out probabilities outside typical set
        logits[~typical_mask] = -float('inf')

        return logits

    def beam_search(
        self,
        prompt_tokens: torch.Tensor,
        num_beams: int = 5,
        max_length: int = 100,
        length_penalty: float = 1.0,
        early_stopping: bool = True
    ) -> Dict[str, torch.Tensor]:
        """Generate using beam search."""
        batch_size = prompt_tokens.size(0)
        device = prompt_tokens.device
        vocab_size = self.model.config.vocab_size

        # Initialize beam search
        beam_scores = torch.zeros((batch_size, num_beams), device=device)
        beam_tokens = prompt_tokens.unsqueeze(1).repeat(1, num_beams, 1)
        beam_tokens = beam_tokens.view(batch_size * num_beams, -1)

        # Track finished sequences
        done = torch.zeros(batch_size, dtype=torch.bool, device=device)

        for step in range(max_length):
            # Get model predictions
            logits, _ = self.model(beam_tokens)
            next_token_logits = logits[:, -1, :]  # [batch_size * num_beams, vocab_size]

            # Calculate log probabilities
            next_token_scores = F.log_softmax(next_token_logits, dim=-1)
            next_token_scores = next_token_scores.view(batch_size, num_beams, vocab_size)

            # Add to beam scores
            next_token_scores = beam_scores.unsqueeze(-1) + next_token_scores

            # Reshape for top-k selection
            next_token_scores = next_token_scores.view(batch_size, num_beams * vocab_size)

            # Select top-k candidates
            next_token_scores, next_tokens = torch.topk(
                next_token_scores, 2 * num_beams, dim=1, largest=True, sorted=True
            )

            # Calculate which beam each token came from
            next_indices = next_tokens // vocab_size
            next_tokens = next_tokens % vocab_size

            # Update beams
            beam_outputs = []
            beam_scores_new = []

            for batch_idx in range(batch_size):
                if done[batch_idx]:
                    # Keep existing beams for finished sequences
                    beam_outputs.append(beam_tokens[batch_idx * num_beams:(batch_idx + 1) * num_beams])
                    beam_scores_new.append(beam_scores[batch_idx])
                    continue

                # Select best beams for this batch
                batch_beam_tokens = []
                batch_beam_scores = []

                for beam_idx in range(num_beams):
                    candidate_idx = batch_idx * num_beams + next_indices[batch_idx, beam_idx]
                    token_id = next_tokens[batch_idx, beam_idx]
                    score = next_token_scores[batch_idx, beam_idx]

                    # Get previous sequence
                    prev_tokens = beam_tokens[candidate_idx // num_beams * num_beams + candidate_idx % num_beams]
                    new_tokens = torch.cat([prev_tokens, token_id.unsqueeze(0)])

                    batch_beam_tokens.append(new_tokens)
                    batch_beam_scores.append(score)

                beam_outputs.append(torch.stack(batch_beam_tokens))
                beam_scores_new.append(torch.stack(batch_beam_scores))

            # Update beam tokens and scores
            beam_tokens = torch.cat([b for b in beam_outputs])
            beam_scores = torch.stack(beam_scores_new)

            # Apply length penalty
            if length_penalty != 1.0:
                length_penalty_factor = ((5.0 + step + 1) / 6.0) ** length_penalty
                beam_scores = beam_scores / length_penalty_factor

        # Return best sequence for each batch item
        best_sequences = []
        for batch_idx in range(batch_size):
            best_beam_idx = torch.argmax(beam_scores[batch_idx])
            best_seq = beam_tokens[batch_idx * num_beams + best_beam_idx]
            best_sequences.append(best_seq)

        return {
            "sequences": torch.stack(best_sequences),
            "scores": beam_scores.max(dim=1)[0]
        }

    def conditional_generate(
        self,
        prompt_tokens: torch.Tensor,
        condition_fn: Callable[[torch.Tensor], torch.Tensor],
        config: GenerationConfig
    ) -> torch.Tensor:
        """Generate sequences conditioned on a custom function."""
        # This allows for custom conditioning logic
        # condition_fn should take logits and return modified logits

        batch_size = prompt_tokens.size(0)
        generated = prompt_tokens.clone()

        for step in range(config.max_length):
            logits, _ = self.model(generated)
            next_token_logits = logits[:, -1, :] / config.temperature

            # Apply custom conditioning
            next_token_logits = condition_fn(next_token_logits)

            # Sample
            if config.do_sample:
                next_token = self._sample_token(next_token_logits, config)
            else:
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

            generated = torch.cat([generated, next_token], dim=-1)

            if generated.size(1) > self.model.config.max_seq_length:
                generated = generated[:, -self.model.config.max_seq_length:]

        return generated


class LOBInference:
    """High-level inference interface for LOB Transformer."""

    def __init__(self, model: LOBTransformer, tokenizer: LOBTokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.generator = LOBGenerator(model, tokenizer)

    def predict_next_events(
        self,
        recent_events: List[TokenComponents],
        n_predictions: int = 10,
        strategy: str = "sampling"
    ) -> List[TokenComponents]:
        """Predict next events given recent market activity."""
        # Encode recent events
        token_ids = [self.tokenizer.encode_token(event) for event in recent_events]
        prompt_tokens = torch.tensor(token_ids, dtype=torch.long).unsqueeze(0)

        # Configure generation
        config = GenerationConfig(
            max_length=n_predictions,
            temperature=0.8,
            top_k=50,
            do_sample=(strategy == "sampling")
        )

        # Generate
        if strategy == "beam":
            output = self.generator.beam_search(prompt_tokens, num_beams=5, max_length=n_predictions)
            generated_ids = output["sequences"][0]
        else:
            output = self.generator.generate(prompt_tokens, config)
            generated_ids = output["sequences"][0]

        # Decode predictions
        start_idx = len(token_ids)
        predicted_tokens = generated_ids[start_idx:]

        predicted_events = []
        for token_id in predicted_tokens:
            try:
                event = self.tokenizer.decode_token(token_id.item())
                if event:
                    predicted_events.append(event)
            except:
                continue

        return predicted_events

    def estimate_probabilities(
        self,
        recent_events: List[TokenComponents],
        candidate_events: List[TokenComponents]
    ) -> Dict[TokenComponents, float]:
        """Estimate probabilities for candidate next events."""
        # Encode recent events
        token_ids = [self.tokenizer.encode_token(event) for event in recent_events]
        prompt_tokens = torch.tensor(token_ids, dtype=torch.long).unsqueeze(0)

        # Get model predictions
        with torch.no_grad():
            logits, _ = self.model(prompt_tokens)
            next_token_logits = logits[0, -1, :]  # Last token predictions
            probs = F.softmax(next_token_logits, dim=-1)

        # Get probabilities for candidate events
        event_probs = {}
        for event in candidate_events:
            try:
                token_id = self.tokenizer.encode_token(event)
                prob = probs[token_id].item()
                event_probs[event] = prob
            except:
                event_probs[event] = 0.0

        return event_probs

    def simulate_market_scenario(
        self,
        initial_events: List[TokenComponents],
        n_steps: int = 100,
        branching_factor: int = 3
    ) -> List[List[TokenComponents]]:
        """Simulate multiple market scenarios."""
        scenarios = []

        for _ in range(branching_factor):
            # Generate one scenario
            predicted_events = self.predict_next_events(
                initial_events,
                n_predictions=n_steps,
                strategy="sampling"
            )

            scenario = initial_events + predicted_events
            scenarios.append(scenario)

        return scenarios
