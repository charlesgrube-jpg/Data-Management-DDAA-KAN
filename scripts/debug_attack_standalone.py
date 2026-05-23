
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from pathlib import Path

# Setup paths
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent.resolve()

sys.path.insert(0, str(PROJECT_ROOT)) # For pipeline
sys.path.insert(0, str(SCRIPT_DIR))   # For train_pinn

print(f"Debug: sys.path includes: {sys.path[:3]}")
print(f"Debug: PROJECT_ROOT = {PROJECT_ROOT}")

# DEBUG: List all files in PROJECT_ROOT to see if 'pipeline' is there
try:
    print(f"CONTENTS OF {PROJECT_ROOT}:")
    print(os.listdir(PROJECT_ROOT))
except Exception as e:
    print(f"Error listing root: {e}")

# Check for file existence
pinn_in_scripts = (SCRIPT_DIR / "train_pinn.py").exists()
pinn_in_root = (PROJECT_ROOT / "train_pinn.py").exists()
print(f"Debug: train_pinn.py in scripts/? {pinn_in_scripts}")
print(f"Debug: train_pinn.py in root/? {pinn_in_root}")

# Import Model - Direct import to fail reliably if missing
try:
    from train_pinn import PINNDetector
except ImportError as e:
    print(f"Import failed directly: {e}")
    # Try importing from scripts package as fallback if scripts is a package
    try:
        from scripts.train_pinn import PINNDetector
    except ImportError as e2:
        print(f"Import failed from scripts: {e2}")
        raise e

# Import Attack
from pipeline.attacks import ImperceptibleAttack

def run_debug():
    print("=== DEBUGGING ATTACK ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 1. Load Model
    print("\n[1] Loading Model...")
    checkpoint_path = PROJECT_ROOT / "models_pinn/pinn_best.pt"
    
    if not checkpoint_path.exists():
        print(f"Error: {checkpoint_path} not found.")
        print(f"Contents of {PROJECT_ROOT}:")
        print(os.listdir(PROJECT_ROOT))
        models_dir = PROJECT_ROOT / "models_pinn"
        if models_dir.exists():
             print(f"Contents of {models_dir}:")
             print(os.listdir(models_dir))
        else:
             # Try checking older locations or generic 'checkpoints'
             print(f"{models_dir} does not exist. Checking 'checkpoints'...")
             chk_dir = PROJECT_ROOT / "checkpoints"
             if chk_dir.exists():
                 print(os.listdir(chk_dir))
        return

    model = PINNDetector()
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    try:
        model.load_state_dict(checkpoint["model_state_dict"])
    except:
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model.to(device)
    model.eval()
    print("Model loaded.")

    # 2. Create Dummy Input (or load real if possible)
    # Let's create a synthetic sine wave to be reproducible
    print("\n[2] Creating Dummy Input...")
    sr = 16000
    duration = 4.0
    t = torch.linspace(0, duration, int(sr * duration))
    # A simple sine wave
    wav = torch.sin(2 * torch.pi * 440 * t) 
    wav = wav.unsqueeze(0).to(device) # (1, Time)
    print(f"Input shape: {wav.shape}")

    # 3. Check Clean Prediction
    print("\n[3] Clean Prediction...")
    with torch.no_grad():
        logits = model(wav)
        probs = F.softmax(logits, dim=-1)
        pred = logits.argmax(dim=-1).item()
    
    print(f"Clean Logits: {logits.detach().cpu().numpy()}")
    print(f"Clean Probs: {probs.detach().cpu().numpy()}")
    print(f"Clean Prediction: {pred}")

    # 4. Setup Attack
    # We want to target the OPPOSITE of the clean prediction
    # to force the attack to work.
    target_label = 1 - pred
    print(f"\n[4] Setting up Attack -> Target: {target_label}")

    attack = ImperceptibleAttack(
        epsilon=0.01,
        alpha_init=0.002,
        max_iterations=50,
        use_room_simulation=False,
    )

    # 5. Run Manual Optimization Step (Trace)
    print("\n[5] Running Manual Trace of Attack Step...")
    
    # Init Delta
    delta = torch.zeros_like(wav, requires_grad=True)
    optimizer = torch.optim.Adam([delta], lr=0.01)
    
    target_tensor = torch.tensor([target_label], device=device)
    
    for i in range(5):
        print(f"\n--- Step {i+1} ---")
        
        adv_wav = wav + delta
        logits = model(adv_wav)
        
        loss = F.cross_entropy(logits, target_tensor)
        
        print(f"Logits: {logits.detach().cpu().numpy()}")
        print(f"Target Label: {target_label}")
        print(f"CE Loss: {loss.item()}")
        
        # Check Gradient Direction
        optimizer.zero_grad()
        loss.backward()
        
        grad_norm = delta.grad.norm().item()
        print(f"Gradient Norm: {grad_norm}")
        
        # Manually update
        optimizer.step()
        
        # Check if logits moved TOWARDS target
        # If target is 0, logits[0] should increase relative to logits[1]
        diff = logits[0, target_label].item() - logits[0, 1-target_label].item()
        print(f"Logit Diff (Target - NonTarget): {diff:.4f}")

    print("\n=== DEBUG COMPLETE ===")

if __name__ == "__main__":
    run_debug()
