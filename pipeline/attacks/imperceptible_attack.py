"""
Imperceptible and Robust Adversarial Attack

Implements the attack from "Imperceptible, Robust, and Targeted Adversarial 
Examples for Automatic Speech Recognition" adapted for deepfake detection.

Loss function (Equation 10):
    ℓ(x, δ, y) = E_t[ℓ_net(f(t(x + δ)), y)] + α · ℓ_θ(x, δ)

Where:
- ℓ_net: Cross-entropy loss against the target model
- t: Random room transformation (EOT)
- ℓ_θ: Psychoacoustic imperceptibility loss
- α: Adaptive weighting between robustness and imperceptibility
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Callable, Dict, Any
from .psychoacoustic_masking import compute_masking_threshold, ImperceptibilityLoss
from .room_simulation import EOTTransformations


class ImperceptibleAttack:
    """
    Imperceptible and robust adversarial attack for audio.
    
    Generates adversarial perturbations that:
    1. Fool the target detector under various room conditions (robust)
    2. Remain below the psychoacoustic masking threshold (imperceptible)
    
    Uses gradient descent with adaptive α to balance these objectives.
    """
    
    def __init__(
        self,
        epsilon: float = 0.01,
        alpha_init: float = 0.5,
        alpha_min: float = 0.01,
        alpha_max: float = 10.0,
        alpha_adjust_rate: float = 1.5,
        num_eot_samples: int = 10,
        max_iterations: int = 100,
        learning_rate: float = 0.001,
        sample_rate: int = 16000,
        n_fft: int = 2048,
        hop_length: int = 512,
        success_threshold: float = 0.8,
        use_room_simulation: bool = True,
        device: str = "auto"
    ):
        """
        Args:
            epsilon: Maximum L-infinity perturbation bound
            alpha_init: Initial imperceptibility weight
            alpha_min: Minimum alpha value
            alpha_max: Maximum alpha value  
            alpha_adjust_rate: Factor to adjust alpha by
            num_eot_samples: Number of EOT transformation samples
            max_iterations: Maximum optimization iterations
            learning_rate: Gradient descent learning rate
            sample_rate: Audio sample rate
            n_fft: FFT size for psychoacoustic model
            hop_length: STFT hop length
            success_threshold: Fraction of EOT samples that must be fooled
            use_room_simulation: Whether to use room transformations
            device: Computation device
        """
        self.epsilon = epsilon
        self.alpha = alpha_init
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.alpha_adjust_rate = alpha_adjust_rate
        self.num_eot_samples = num_eot_samples
        self.max_iterations = max_iterations
        self.learning_rate = learning_rate
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.success_threshold = success_threshold
        self.use_room_simulation = use_room_simulation
        
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # Initialize loss components
        self.imperceptibility_loss = ImperceptibilityLoss(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length
        )
        
        if use_room_simulation:
            self.eot = EOTTransformations(sample_rate=sample_rate)
        else:
            self.eot = None
    
    def attack(
        self,
        model: nn.Module,
        audio: torch.Tensor,
        target_label: int,
        feature_extractor: Optional[Callable] = None,
        initial_perturbation: Optional[torch.Tensor] = None,
        verbose: bool = True
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Generate imperceptible adversarial perturbation.
        
        Args:
            model: Target detector model (takes features, outputs logits)
            audio: Clean audio waveform (samples,) or (batch, samples)
            target_label: Target class to fool model into predicting
            feature_extractor: Function to extract features from audio
                              (e.g., CQT extraction). If None, assumes model
                              takes raw waveform.
            initial_perturbation: Optional initial perturbation (e.g., from 
                                  robust-only attack)
            verbose: Print progress
            
        Returns:
            Adversarial perturbation δ, attack info dictionary
        """
        model.eval()
        model.to(self.device)
        
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
        audio = audio.to(self.device)
        
        batch_size = audio.shape[0]
        
        # Initialize perturbation
        if initial_perturbation is not None:
            delta = initial_perturbation.clone().to(self.device)
        else:
            delta = torch.zeros_like(audio)
        delta.requires_grad = True
        
        # Precompute masking threshold for clean audio
        masking_threshold = compute_masking_threshold(
            audio.squeeze(0) if batch_size == 1 else audio,
            self.sample_rate,
            self.n_fft,
            self.hop_length
        )
        
        # Target label tensor
        target = torch.tensor([target_label] * batch_size, device=self.device)
        
        # Optimizer
        optimizer = torch.optim.Adam([delta], lr=self.learning_rate)
        
        # Track statistics
        history = {
            'total_loss': [],
            'network_loss': [],
            'imperceptibility_loss': [],
            'alpha': [],
            'success_rate': [],
            'snr': []
        }
        
        best_delta = None
        best_success_rate = 0
        
        alpha = self.alpha
        
        for iteration in range(self.max_iterations):
            optimizer.zero_grad()
            
            # Apply perturbation (clamped to epsilon ball)
            delta_clamped = torch.clamp(delta, -self.epsilon, self.epsilon)
            adv_audio = audio + delta_clamped
            
            # Clamp to valid audio range
            adv_audio = torch.clamp(adv_audio, -1.0, 1.0)
            
            # Sample EOT transformations
            if self.eot is not None:
                transforms = self.eot.sample_transformations(self.num_eot_samples)
            else:
                transforms = [lambda x: x]  # Identity if no EOT
            
            # Compute network loss over transformations (EOT)
            network_losses = []
            successes = []
            
            for transform in transforms:
                # Apply room transformation
                transformed = transform(adv_audio)
                
                # Extract features if extractor provided
                if feature_extractor is not None:
                    # Handle batched feature extraction
                    if batch_size == 1:
                        features = feature_extractor(transformed.squeeze(0).cpu().numpy())
                        features = torch.from_numpy(features).unsqueeze(0).to(self.device)
                    else:
                        features_list = [
                            feature_extractor(t.cpu().numpy()) 
                            for t in transformed
                        ]
                        features = torch.stack([
                            torch.from_numpy(f) for f in features_list
                        ]).to(self.device)
                else:
                    features = transformed
                
                # Forward through model
                with torch.enable_grad():
                    logits = model(features)
                    loss = F.cross_entropy(logits, target)
                    network_losses.append(loss)
                
                # Check success
                pred = logits.argmax(dim=-1)
                successes.append((pred == target).float().mean().item())
            
            # Average network loss over EOT samples
            network_loss = torch.stack(network_losses).mean()
            success_rate = sum(successes) / len(successes)
            
            # Compute imperceptibility loss
            imperc_loss = self.imperceptibility_loss(
                audio,
                delta_clamped,
                masking_threshold
            )
            
            # Total loss (Equation 10)
            total_loss = network_loss + alpha * imperc_loss
            
            # Backward and update
            total_loss.backward()
            optimizer.step()
            
            # Adaptive alpha adjustment
            if success_rate >= self.success_threshold:
                # Attack is working, increase focus on imperceptibility
                alpha = min(alpha * self.alpha_adjust_rate, self.alpha_max)
            else:
                # Attack failing, decrease imperceptibility weight
                alpha = max(alpha / self.alpha_adjust_rate, self.alpha_min)
            
            # Compute SNR
            with torch.no_grad():
                signal_power = (audio ** 2).mean()
                noise_power = (delta_clamped ** 2).mean()
                snr = 10 * torch.log10(signal_power / (noise_power + 1e-10))
            
            # Track best result
            if success_rate > best_success_rate:
                best_success_rate = success_rate
                best_delta = delta_clamped.detach().clone()
            
            # Log history
            history['total_loss'].append(total_loss.item())
            history['network_loss'].append(network_loss.item())
            history['imperceptibility_loss'].append(imperc_loss.item())
            history['alpha'].append(alpha)
            history['success_rate'].append(success_rate)
            history['snr'].append(snr.item())
            
            if verbose and (iteration + 1) % 10 == 0:
                print(f"Iter {iteration+1:3d} | "
                      f"Loss: {total_loss.item():.4f} | "
                      f"Net: {network_loss.item():.4f} | "
                      f"Imp: {imperc_loss.item():.4f} | "
                      f"α: {alpha:.3f} | "
                      f"SR: {success_rate:.2f} | "
                      f"SNR: {snr.item():.1f}dB")
        
        # Return best perturbation found
        if best_delta is None:
            best_delta = delta_clamped.detach()
        
        info = {
            'success_rate': best_success_rate,
            'snr': history['snr'][-1],
            'final_alpha': alpha,
            'iterations': self.max_iterations,
            'history': history
        }
        
        return best_delta, info
    
    def targeted_attack(
        self,
        model: nn.Module,
        audio: torch.Tensor,
        original_label: int,
        target_label: int,
        feature_extractor: Optional[Callable] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Targeted attack: fool model into predicting specific class.
        
        Args:
            model: Target detector
            audio: Clean audio
            original_label: True label
            target_label: Desired adversarial label
            feature_extractor: Feature extraction function
            
        Returns:
            Adversarial perturbation, attack info
        """
        return self.attack(model, audio, target_label, feature_extractor, **kwargs)
    
    def untargeted_attack(
        self,
        model: nn.Module,
        audio: torch.Tensor,
        original_label: int,
        feature_extractor: Optional[Callable] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Untargeted attack: fool model into predicting any wrong class.
        
        For binary classification (real/fake), this is equivalent to
        targeted attack with the opposite label.
        
        Args:
            model: Target detector  
            audio: Clean audio
            original_label: True label
            feature_extractor: Feature extraction function
            
        Returns:
            Adversarial perturbation, attack info
        """
        # For binary classification, target the opposite class
        target_label = 1 - original_label
        return self.attack(model, audio, target_label, feature_extractor, **kwargs)


class PGDAttack:
    """
    Standard PGD attack baseline (without psychoacoustic masking).
    
    For comparison with imperceptible attack.
    """
    
    def __init__(
        self,
        epsilon: float = 0.01,
        alpha: float = 0.001,
        num_steps: int = 40,
        random_start: bool = True,
        device: str = "auto"
    ):
        self.epsilon = epsilon
        self.alpha = alpha
        self.num_steps = num_steps
        self.random_start = random_start
        
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
    
    def attack(
        self,
        model: nn.Module,
        audio: torch.Tensor,
        target_label: int,
        feature_extractor: Optional[Callable] = None
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Standard PGD attack."""
        model.eval()
        model.to(self.device)
        
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
        audio = audio.to(self.device)
        
        # Initialize
        if self.random_start:
            delta = torch.empty_like(audio).uniform_(-self.epsilon, self.epsilon)
        else:
            delta = torch.zeros_like(audio)
        
        target = torch.tensor([target_label], device=self.device)
        
        for _ in range(self.num_steps):
            delta.requires_grad = True
            
            adv_audio = audio + delta
            adv_audio = torch.clamp(adv_audio, -1.0, 1.0)
            
            if feature_extractor is not None:
                features = feature_extractor(adv_audio.squeeze(0).cpu().numpy())
                features = torch.from_numpy(features).unsqueeze(0).to(self.device)
            else:
                features = adv_audio
            
            logits = model(features)
            loss = F.cross_entropy(logits, target)
            
            loss.backward()
            
            # PGD step
            delta = delta + self.alpha * delta.grad.sign()
            delta = torch.clamp(delta, -self.epsilon, self.epsilon)
            delta = delta.detach()
        
        # Check success
        with torch.no_grad():
            adv_audio = audio + delta
            if feature_extractor is not None:
                features = feature_extractor(adv_audio.squeeze(0).cpu().numpy())
                features = torch.from_numpy(features).unsqueeze(0).to(self.device)
            else:
                features = adv_audio
            logits = model(features)
            pred = logits.argmax(dim=-1)
            success = (pred == target).item()
        
        info = {
            'success': success,
            'epsilon': self.epsilon,
            'num_steps': self.num_steps
        }
        
        return delta, info


if __name__ == "__main__":
    # Test attack setup (without actual model)
    print("Imperceptible Attack Configuration:")
    attack = ImperceptibleAttack(
        epsilon=0.01,
        num_eot_samples=5,
        max_iterations=50
    )
    print(f"  Epsilon: {attack.epsilon}")
    print(f"  EOT samples: {attack.num_eot_samples}")
    print(f"  Max iterations: {attack.max_iterations}")
    print(f"  Device: {attack.device}")
    print("\nTo run attack, provide a trained model and audio sample.")
