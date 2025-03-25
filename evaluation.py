import torch
from tqdm import tqdm
from utils import get_device
import numpy as np
import scipy.stats as stats

def evaluate_model(model, test_loader, device=None):
    if device is None:
        device = get_device()
    
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    return accuracy

def evaluate_attack(model, test_loader, attack_fn, eps=0.05, device=None):
    if device is None:
        device = get_device()
    
    model.eval()
    successful_attacks = []
    total_attacks = 0

    for images, labels in tqdm(test_loader, desc="Evaluating attack"):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, dim=1)

        mask = (predicted == labels)
        if mask.sum().item() == 0:
            continue

        images = images[mask]
        labels = labels[mask]
        total_attacks += images.size(0)

        adv_images = attack_fn(model, images, labels, eps=eps)
        outputs_adv = model(adv_images)
        _, predicted_adv = torch.max(outputs_adv, dim=1)

        successful_attacks.extend((predicted_adv != labels).cpu().numpy())

    asr = 100 * np.mean(successful_attacks) if successful_attacks else 0
    return asr, successful_attacks

def compute_confidence_interval(data, confidence=0.95):
    data = np.array(data)
    if len(data) == 0:
        return 0, 0

    mean = np.mean(data)
    std_err = np.std(data, ddof=1) / np.sqrt(len(data))

    alpha = 1 - confidence
    z_score = stats.norm.ppf(1 - alpha / 2)
    margin = std_err * z_score
    return mean - margin, mean + margin
