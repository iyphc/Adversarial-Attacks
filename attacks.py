import torch.nn as nn
import torch
import torch.nn.functional as F
from train import get_device
from model import LittleCNN

def DeepFool(model, image, num_classes=10, overshoot=0.02, max_iter=50, device='cpu'):
    device = get_device()
    image = image.to(device)
    image = image.clone().detach()

    output = model(image.unsqueeze(0))
    _, label = torch.max(output, 1)
    label = label.item()

    perturbed_image = image.clone()
    r_tot = torch.zeros_like(image)
    i = 0

    # DeefFool
    while i < max_iter:
        perturbed_image.requires.grad = True
        output = model(perturbed_image)
        fs = output[0]
        cur_label = fs.argmax().item()
        if cur_label != label:
            break

        model.zero_grad()
        fs[label].backward(retain_graph=True)
        grad_orig = perturbed_image.grad.detach().clone()
        perturbed_image.grad.zero_()

        min_perturbation = float('inf')
        w = None

        for k in range(num_classes):
            if (k == label):
                continue
            fs[k].backward(retain_graph=True)
            cur_grad = perturbed_image.grad.detach().clone()
            perturbed_image.grad.zero_()

            w_k = cur_grad - grad_orig
            f_k = (fs[k] - fs[label]).detach()
            norm_w = w_k.view(-1).norm()
            pert_k = torch.abs(f_k) / norm_w
            if (pert_k < min_perturbation):
                min_perturbation = pert_k
                w = w_k
        r_i = (min_perturbation / norm_w) * w
        r_tot += r_i 
        perturbed_image = torch.clamp(image + r_tot, 0, 1).detach()
        i+=1
    return perturbed_image

