
import torch
import numpy as np

def evaluate_adversarial_robustness_cleverhans(model, dataloader, attack, device, num_samples):
    results = {
        'total_samples': 0, 'successful_attacks': 0, 'snr_values': [],
        'clean_correct': 0, 'adversarial_correct': 0
    }
    
    for i, (audio, label) in enumerate(dataloader):
        if i >= num_samples: break
        
        audio = audio.to(device) # (1, Length)
        label = label.to(device)
        
        # Clean Pred
        with torch.no_grad():
            clean_logits = model(audio)
            clean_pred = clean_logits.argmax(dim=-1)
        
        if clean_pred == label:
            results['clean_correct'] += 1
        
        # Attack
        target_label = 1 - label # Target opposite class
        
        # Run CleverHans Optimized Attack
        delta = attack.attack(audio, target_label)
        adv_audio = torch.clamp(audio + delta, -1.0, 1.0)
        
        # Adv Pred
        with torch.no_grad():
            adv_logits = model(adv_audio)
            adv_pred = adv_logits.argmax(dim=-1)
        
        if adv_pred == target_label:
            results['successful_attacks'] += 1
        if adv_pred == label:
            results['adversarial_correct'] += 1
            
        # SNR
        noise_power = (delta ** 2).mean()
        signal_power = (audio ** 2).mean()
        snr = 10 * torch.log10(signal_power / (noise_power + 1e-10))
        results['snr_values'].append(snr.item())
        results['total_samples'] += 1
        
        print(f"Sample {i}: Clean {clean_pred.item()} | Adv {adv_pred.item()} | SNR {snr.item():.2f} dB")
        
    results['attack_success_rate'] = results['successful_attacks'] / max(results['total_samples'], 1)
    results['clean_accuracy'] = results['clean_correct'] / max(results['total_samples'], 1)
    results['adversarial_accuracy'] = results['adversarial_correct'] / max(results['total_samples'], 1)
    results['mean_snr'] = np.mean(results['snr_values']) if results['snr_values'] else 0.0
    
    return results
