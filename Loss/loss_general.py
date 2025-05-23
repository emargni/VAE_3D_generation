import torch
import torch.nn.functional as F

def loss_bce_kld(
    recon_x: torch.Tensor,
    x: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    input_dim: int
) -> torch.Tensor:
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, input_dim), reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

def loss_mse_kld(
    recon_x: torch.Tensor,
    x: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    input_dim: int
) -> torch.Tensor:
    MSE = F.mse_loss(recon_x, x.view(-1, input_dim), reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return MSE + KLD

def loss_dice_kld(
    recon_x: torch.Tensor,
    x: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    input_dim: int,
    epsilon: float = 1e-6
) -> torch.Tensor:
    recon = recon_x.view(-1, input_dim)
    target = x.view(-1, input_dim)
    intersection = (recon * target).sum()
    dice = 1 - (2. * intersection + epsilon) / (recon.sum() + target.sum() + epsilon)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return dice + KLD

def loss_focal_kld(
    recon_x: torch.Tensor,
    x: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    input_dim: int,
    gamma: float = 2.0,
    alpha: float = 0.25
) -> torch.Tensor:
    recon = recon_x.view(-1, input_dim)
    target = x.view(-1, input_dim)
    BCE = F.binary_cross_entropy(recon, target, reduction='none')
    focal = alpha * (1 - recon) ** gamma * BCE
    focal_loss = focal.sum()
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return focal_loss + KLD

def loss_iou_kld(
    recon_x: torch.Tensor,
    x: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    input_dim: int,
    epsilon: float = 1e-6
) -> torch.Tensor:
    recon = recon_x.view(-1, input_dim)
    target = x.view(-1, input_dim)
    intersection = (recon * target).sum()
    union = recon.sum() + target.sum() - intersection
    iou = 1 - (intersection + epsilon) / (union + epsilon)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return iou + KLD
