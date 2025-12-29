import torch
import torch.nn as nn

class HeteroscedasticLoss(nn.Module):
    def forward(self, preds, targets):
        # preds: (Batch, 2, H, W) -> Channel 0: Mean, Channel 1: LogVar
        # targets: (Batch, 1, H, W)
        
        mu = preds[:, 0:1, :, :]
        log_var = preds[:, 1:2, :, :]
        
        # Clamp log_var untuk stabilitas numerik
        log_var = torch.clamp(log_var, min=-10, max=10)
        
        precision = torch.exp(-log_var)
        mse = (mu - targets)**2
        loss = 0.5 * (precision * mse + log_var)
        
        # Masking background (NoData)
        mask = targets > 0
        if mask.sum() > 0:
            return loss[mask].mean()
        return loss.mean()
