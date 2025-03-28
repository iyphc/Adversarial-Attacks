import torch.nn as nn
import torch
import torch.nn.functional as F
from utils import get_device
from model import LittleCNN

def DeepFool(model, image, num_classes=10, overshoot=0.02, eps=0.05, max_iter=50, device=None):
    if device is None:
        device = get_device()
    model.eval()
    image = image.to(device)
    image = image.clone().detach()

    output = model(image.unsqueeze(0))
    _, label = torch.max(output, 1)
    label = label.item()

    perturbed_image = image.clone()
    r_tot = torch.zeros_like(image)
    i = 0

    while i < max_iter:
        perturbed_image.requires_grad = True
        output = model(perturbed_image.unsqueeze(0))
        fs = output[0]
        current_label = fs.argmax().item()
        
        if current_label != label:
            break

        model.zero_grad()
        fs[label].backward(retain_graph=True)
        grad_orig = perturbed_image.grad.detach().clone()
        perturbed_image.grad.zero_()

        min_pert = float('inf')
        w = None

        for k in range(num_classes):
            if k == label:
                continue
                
            fs[k].backward(retain_graph=True)
            curr_grad = perturbed_image.grad.detach().clone()
            perturbed_image.grad.zero_()
            
            w_k = curr_grad - grad_orig
            f_k = (fs[k] - fs[label]).detach()
            
            norm_w = w_k.view(-1).norm()
            if norm_w == 0:
                continue
                
            pert_k = torch.abs(f_k) / norm_w
            if pert_k < min_pert:
                min_pert = pert_k
                w = w_k

        if w is None:
            break

        r_i = (min_pert / norm_w) * w * (1 + overshoot)
        r_tot += r_i.squeeze()

        current_norm = torch.abs(r_tot).max().item()

        if current_norm > eps:
            scale = eps / current_norm
            r_tot = r_tot * scale
            perturbed_image = torch.clamp(image + r_tot, 0, 1).detach()
            break

        perturbed_image = torch.clamp(image + r_tot, 0, 1).detach()
        i += 1

    return perturbed_image

def PGD(model, image, num_classes=10, alpha=0.005, eps=0.05, max_iter=50, device=None):
    if device is None:
        device = get_device()
    model.eval()
    image = image.to(device)
    image = image.clone().detach()

    output = model(image.unsqueeze(0))
    _, label = torch.max(output, 1)
    label = label.item()

    perturbed_image = image.clone().detach().requires_grad_(True)

    for _ in range(max_iter):
        output = model(perturbed_image.unsqueeze(0))

        _, adv_label = torch.max(output, 1)
        adv_label = adv_label.item()
        
        if (label != adv_label):
            break

        loss = nn.CrossEntropyLoss()(output, torch.tensor([label], device=device))
        
        model.zero_grad()
        loss.backward()
        
        grad_sign = perturbed_image.grad.data.sign()
        perturbed_image = perturbed_image + alpha * grad_sign
        perturbed_image = torch.min(torch.max(perturbed_image, image - eps), image + eps)
        perturbed_image = torch.clamp(perturbed_image, 0, 1).detach().requires_grad_(True)


    return perturbed_image
