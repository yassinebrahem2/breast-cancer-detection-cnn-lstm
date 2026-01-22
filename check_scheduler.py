"""
Verify Learning Rate Scheduler behavior from checkpoint
"""
import torch
import json

checkpoint_path = "outputs/checkpoint_epoch_020.pth"
config_path = "outputs/train_config.json"

print("Loading checkpoint...")
checkpoint = torch.load(checkpoint_path, map_location='cpu')

print(f"\nCheckpoint info:")
print(f"  Epoch: {checkpoint['epoch']}")
print(f"  Val Loss: {checkpoint['val_loss']:.4f}")

# Check scheduler state
if 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict']:
    scheduler_state = checkpoint['scheduler_state_dict']
    print(f"\nScheduler state:")
    print(f"  Best: {scheduler_state.get('best', 'N/A')}")
    print(f"  Num bad epochs: {scheduler_state.get('num_bad_epochs', 'N/A')}")
    print(f"  Cooldown counter: {scheduler_state.get('cooldown_counter', 'N/A')}")
    
    # Check optimizer LR
    if 'optimizer_state_dict' in checkpoint:
        optimizer_state = checkpoint['optimizer_state_dict']
        if 'param_groups' in optimizer_state:
            for i, pg in enumerate(optimizer_state['param_groups']):
                print(f"\nOptimizer param_group {i}:")
                print(f"  LR: {pg['lr']}")
                print(f"  Weight decay: {pg.get('weight_decay', 'N/A')}")
else:
    print("\n⚠️  No scheduler state found in checkpoint!")

# Load training history
print(f"\n\nValidation loss history:")
if 'history' in checkpoint and 'val_loss' in checkpoint['history']:
    val_losses = checkpoint['history']['val_loss']
    print(f"  Total epochs: {len(val_losses)}")
    print(f"  Best val loss: {min(val_losses):.4f} at epoch {val_losses.index(min(val_losses)) + 1}")
    
    # Check for plateau detection (3 epochs without improvement)
    print(f"\n  Last 10 epochs val loss:")
    for i, loss in enumerate(val_losses[-10:], start=len(val_losses)-9):
        marker = "  ← BEST" if loss == min(val_losses) else ""
        print(f"    Epoch {i}: {loss:.4f}{marker}")
    
    # Analyze plateaus
    print(f"\n  Plateau analysis (patience=3):")
    best_so_far = float('inf')
    bad_epochs = 0
    for i, loss in enumerate(val_losses, start=1):
        if loss < best_so_far:
            best_so_far = loss
            bad_epochs = 0
            print(f"    Epoch {i}: NEW BEST ({loss:.4f}) - Counter reset")
        else:
            bad_epochs += 1
            if bad_epochs == 3:
                print(f"    Epoch {i}: PLATEAU DETECTED - Should reduce LR (bad_epochs={bad_epochs})")
                bad_epochs = 0  # Reset after LR reduction
            elif bad_epochs > 3:
                pass  # Don't print every epoch
