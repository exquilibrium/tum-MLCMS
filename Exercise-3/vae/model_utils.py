import torch
from pathlib import Path

def save_model(model: torch.nn.Module,
               target_dir: str,
               model_name: str):
  
  """Saves a PyTorch model to a target directory.

  Args:
    model: A target PyTorch model to save.
    target_dir: A directory for saving the model to.
    model_name: A filename for the saved model. Should include
      either ".pth" or ".pt" as the file extension.

  Example usage:
    save_model(model=model_0,
               target_dir="models",
               model_name="05_going_modular_tingvgg_model.pth")
  """
  # Create target directory
  target_dir_path = Path(target_dir)
  target_dir_path.mkdir(parents=True,
                        exist_ok=True)

  # Create model save path
  assert model_name.endswith(".pth") or model_name.endswith(".pt"), "model_name should end with '.pt' or '.pth'"
  model_save_path = target_dir_path / model_name

  # Save the model state_dict()
  print(f"[INFO] Saving model to: {model_save_path}")
  torch.save(obj=model.state_dict(),
             f=model_save_path)
  

def save_best_model(model, val_loss, best_val_loss, save_path):
    """
    Save the best model based on validation loss.

    Parameters:
    - model: The PyTorch model to be saved.
    - optimizer: The optimizer used for training.
    - val_loss: The current validation loss.
    - best_val_loss: The best validation loss so far.
    - save_path: The file path where the model will be saved.

    Returns:
    - best_val_loss: Updated best validation loss.
    """
    if val_loss < best_val_loss:
        print(f'Test loss improved ({best_val_loss:.6f} --> {val_loss:.6f}). Saving the model...')
        best_val_loss = val_loss
        save_model(model, save_path, 'best_model.pth')

    return best_val_loss