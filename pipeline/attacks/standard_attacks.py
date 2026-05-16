
import torch
import torch.nn as nn
import torch.nn.functional as F

class FGSMAttack:
    """
    Fast Gradient Sign Method (FGSM) Attack.
    Perturbs input by epsilon * sign(gradient) to maximize loss.
    """
    def __init__(self, model, epsilon=0.005, device='cpu'):
        self.model = model
        self.epsilon = epsilon
        self.device = device

    def attack(self, data, target):
        """
        Run FGSM attack.
        data: (Batch, Length) - input audio
        target: (Batch) - target label (for targeted attack) OR true label (for untargeted)
        """
        # Create a clone to avoid modifying original data
        data_adv = data.clone().detach().to(self.device)
        data_adv.requires_grad = True

        # Forward pass
        logits = self.model(data_adv)
        
        # Loss: We want to MAXIMIZE loss w.r.t true label (Untargeted)
        # OR MINIMIZE loss w.r.t target label (Targeted)
        # Standard FGSM is untargeted (maximize error).
        # But our CleverHans attack was TARGETED (make it look like opposite class).
        # Let's support both but default to Untargeted for "Robustness Baseline".
        # Actually, for binary classification, Untargeted (Away from True) == Targeted (Towards False).
        
        loss = F.cross_entropy(logits, target)

        # Backward pass
        self.model.zero_grad()
        loss.backward()

        # Generate perturbation
        # maximizing loss -> move in direction of gradient
        data_grad = data_adv.grad.data
        sign_data_grad = data_grad.sign()
        
        # Create adversarial example
        perturbed_data = data_adv + self.epsilon * sign_data_grad
        
        # Clip to valid range [-1, 1]
        perturbed_data = torch.clamp(perturbed_data, -1.0, 1.0)

        return perturbed_data - data # Return delta


class PGDAttack:
    """
    Projected Gradient Descent (PGD) Attack.
    Iterative version of FGSM with random start and projection.
    """
    def __init__(self, model, epsilon=0.005, alpha=0.001, steps=10, device='cpu'):
        self.model = model
        self.epsilon = epsilon
        self.alpha = alpha
        self.steps = steps
        self.device = device

    def attack(self, data, target):
        """
        Run PGD attack (Untargeted: Maximize Loss).
        """
        # Start with random perturbation within epsilon ball
        delta = torch.zeros_like(data).uniform_(-self.epsilon, self.epsilon).to(self.device)
        delta.requires_grad = True
        
        orig_data = data.to(self.device)

        for _ in range(self.steps):
            # Forward pass
            adv_data = torch.clamp(orig_data + delta, -1.0, 1.0)
            logits = self.model(adv_data)
            
            # Loss: Maximize CrossEntropy with True Label
            loss = F.cross_entropy(logits, target)
            
            # Backward
            self.model.zero_grad()
            loss.backward()
            
            # Update delta
            grad = delta.grad.detach()
            delta.data = delta.data + self.alpha * grad.sign()
            
            # Project data back to epsilon ball
            delta.data = torch.clamp(delta.data, -self.epsilon, self.epsilon)
            
            # Reset gradients
            delta.grad.zero_()

        # Final check to clip to valid audio range
        adv_final = torch.clamp(orig_data + delta, -1.0, 1.0)
        
        return adv_final - orig_data
