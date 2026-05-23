
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from . import generate_masking_threshold as generate_mask

class CleverHansAttack:
    """
    PyTorch implementation of the CleverHans 'Imperceptible, Robust, and Targeted' attack.
    Original TF code: https://github.com/yaq007/cleverhans/tree/master/examples/adversarial_asr
    
    This port maintains the exact two-stage optimization logic:
    Stage 1: Valid adversarial example generation (ASR/Classification success).
    Stage 2: Refinement for imperceptibility (Masking Loss optimization).
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        max_length: int = 80000, # 5 seconds at 16khz
        sample_rate: int = 16000,
        window_size: int = 2048,
        batch_size: int = 1,
        lr_stage1: float = 1.0,  # Adjusted for PyTorch scale (TF used 100/32768 approx)
        lr_stage2: float = 0.1,  # Adjusted
        num_iter_stage1: int = 1000,
        num_iter_stage2: int = 4000,
        initial_bound: float = 0.1 # approx 2000/32768 from TF code
    ):
        self.model = model
        self.device = device
        self.max_length = max_length
        self.sample_rate = sample_rate
        self.window_size = window_size
        self.batch_size = batch_size
        
        self.lr_stage1 = lr_stage1
        self.lr_stage2 = lr_stage2
        self.num_iter_stage1 = num_iter_stage1
        self.num_iter_stage2 = num_iter_stage2
        self.initial_bound = initial_bound

    def compute_masking_threshold(self, x):
        """
        Compute the masking threshold for the input audio.
        x: numpy array (batch, length)
        Returns: threshold (batch, F, T), psd_max (batch)
        """
        th_batch = []
        psd_max_batch = []
        
        for i in range(x.shape[0]):
            audio = x[i]
            # generate_mask expects audio in range [-32768, 32768] usually, 
            # but our code uses [-1, 1]. The original code passed `audio_np` which was * 32768.
            # We should probably scale it up for the psychoacoustic model to work as intended 
            # (since hardcoded constants likely assume 16-bit integer scale).
            audio_scaled = audio * 32768.0
            
            th, psd_max = generate_mask.generate_th(audio_scaled, self.sample_rate, self.window_size)
            th_batch.append(th)
            psd_max_batch.append(psd_max)

        return np.array(th_batch), np.array(psd_max_batch)

    def compute_masking_loss(self, delta, th_batch, psd_max_batch):
        """
        Compute the masking loss: mean(relu(PSD(delta) - threshold))
        delta: (Batch, Time)
        """
        losses = []
        for i in range(delta.shape[0]):
            d = delta[i]
            # Convert delta to same scale as threshold calculation (16-bit)
            d_scaled = d * 32768.0 
            
            # STFT and PSD (mirroring generate_mask.compute_PSD_matrix logic)
            # We implement this in PyTorch for differentiability
            
            window = torch.hann_window(self.window_size).to(self.device)
            stft = torch.stft(
                d_scaled, 
                n_fft=self.window_size, 
                hop_length=self.window_size // 4, # librosa default hop is usually win/4
                window=window,
                center=False,
                return_complex=True
            )
            
            # Magnitude
            z = torch.abs(stft) / self.window_size * np.sqrt(8.0/3.) # normalization factor from original code
            
            psd = 10 * torch.log10(z * z + 1e-20)
            PSD = 96 - torch.tensor(psd_max_batch[i]).to(self.device) + psd
            
            # Loss: ReLU(PSD - Threshold)
            # Threshold shape mismatch? th_batch is (time_frames, freq_bins) usually from the script
            # librosa/scipy outputs (freq, time). 
            # generate_mask returns theta_xs as (time, freq).
            # torch stft returns (freq, time).
            
            th = torch.tensor(th_batch[i]).to(self.device).transpose(0, 1) # (Freq, Time)
            
            # Align dimensions
            # STFT might have slightly different size depending on padding/centering.
            # Original code: librosa.core.stft(center=False)
            # PyTorch stft(center=False) should match if hop/win are same.
            
            min_cols = min(PSD.shape[1], th.shape[1])
            PSD = PSD[:, :min_cols]
            th = th[:, :min_cols]
            
            loss = torch.relu(PSD - th).mean()
            losses.append(loss)
            
        return torch.stack(losses).mean()

    def attack(self, audio, target_label):
        """
        Perform the two-stage attack.
        audio: (Batch, Length) tensor, range [-1, 1]
        target_label: (Batch) tensor
        """
        batch_size = audio.shape[0]
        self.model.eval()
        
        # 0. Precompute Masking Threshold
        audio_np = audio.detach().cpu().numpy()
        th_batch, psd_max_batch = self.compute_masking_threshold(audio_np)
        
        
        # Variables to optimize
        delta = torch.zeros_like(audio, requires_grad=True, device=self.device)
        rescale = torch.ones(batch_size, 1, device=self.device)
        
        # ----------------------
        # STAGE 1: Optimization
        # ----------------------
        print("Starting Stage 1: Adversarial Success...")
        optimizer1 = optim.Adam([delta], lr=self.lr_stage1)
        
        best_delta = torch.zeros_like(audio)
        
        for i in range(self.num_iter_stage1):
            optimizer1.zero_grad()
            
            # Apply delta with clipping and rescaling
            # In Stage 1, we just want ANY success, so we loosely bound it
            d_clamped = torch.clamp(delta, -self.initial_bound, self.initial_bound) * rescale
            adv = torch.clamp(audio + d_clamped, -1.0, 1.0)
            
            logits = self.model(adv)
            loss_ce = F.cross_entropy(logits, target_label)
            
            loss_ce.backward()
            
            # Sign update (FGSM-like in Adam? Original code used tf.sign(grad) with Adam)
            # "self.train1 = self.optimizer1.apply_gradients([(tf.sign(grad1), var1)])"
            # Yes, they used sign descent.
            if delta.grad is not None:
                delta.grad = torch.sign(delta.grad)
            
            optimizer1.step()
            
            # Check success and update rescale
            with torch.no_grad():
                preds = logits.argmax(dim=1)
                for b in range(batch_size):
                    if preds[b] == target_label[b]:
                        # If successful, try to shrink the bound to find minimal perturbation
                        current_max = d_clamped[b].abs().max()
                        if rescale[b] * self.initial_bound > current_max:
                            rescale[b] = current_max / self.initial_bound
                        rescale[b] *= 0.8 # Tighten bound
                        best_delta[b] = d_clamped[b].detach()
            
            if i % 100 == 0:
                print(f"Stage 1 Iter {i}: Loss {loss_ce.item():.4f}")

        # ----------------------
        # STAGE 2: Refinement
        # ----------------------
        print("Starting Stage 2: Imperceptibility Refinement...")
        
        # Init variables with best from Stage 1
        delta = best_delta.clone().detach().requires_grad_(True)
        alpha = torch.ones(batch_size, device=self.device) * 0.05 # Initial alpha
        
        optimizer2 = optim.Adam([delta], lr=self.lr_stage2)
        
        final_deltas = best_delta.clone()
        min_th = 0.0005
        
        for i in range(self.num_iter_stage2):
            optimizer2.zero_grad()
            
            # Adv example
            adv = torch.clamp(audio + delta, -1.0, 1.0) 
            
            logits = self.model(adv)
            loss_ce = F.cross_entropy(logits, target_label, reduction='none') # per sample
            
            # Masking Loss
            loss_th = self.compute_masking_loss(delta, th_batch, psd_max_batch)
            
            # Total loss: CE + alpha * Masking
            # We need to broadcast alpha
            total_loss = loss_ce.mean() + (alpha * loss_th).mean()
            
            total_loss.backward()
            optimizer2.step()
            
            # Check success and Adjust Alpha
            with torch.no_grad():
                preds = logits.argmax(dim=1)
                for b in range(batch_size):
                    if preds[b] == target_label[b]:
                        # Success! We can afford to be quieter.
                        # Save this good example
                        final_deltas[b] = delta[b].detach()
                        
                        if i % 20 == 0:
                            alpha[b] *= 1.2 # Critical: Increase weight on imperceptibility
                    else:
                        # Failed. Need to be louder (focus more on CE loss).
                        if i % 50 == 0:
                            alpha[b] *= 0.8
                            alpha[b] = max(alpha[b], min_th)
            
            if i % 100 == 0:
                 print(f"Stage 2 Iter {i}: CE {loss_ce.mean().item():.4f} | TH {loss_th.item():.4f} | Alpha {alpha.mean().item():.4f}")

        return final_deltas

