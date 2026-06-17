import torch
import numpy as np
from sklearn.base import BaseEstimator

def evaluate_transfer(models, x_adv, y_target):
    results = {}

    for name, model in models.items():
        if isinstance(model, BaseEstimator):
            # scikit-learn model
            preds = model.predict(x_adv)
        elif isinstance(model, torch.nn.Module):
            # PyTorch model
            model.eval()
            with torch.no_grad():
                preds = model(torch.tensor(x_adv, dtype=torch.float32))
                preds = preds.argmax(dim=1).numpy()
        else:
            raise TypeError(f"Unsupported model type for {name}: {type(model)}")

        # Compute success rate (adversarial success)
        success = (preds == y_target).mean()
        results[name] = success

    return results