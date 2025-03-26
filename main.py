from utils import get_device
from train import train_model
from model import LittleCNN
from data import get_dataloaders
from attacks import PGD, DeepFool
from evaluation import evaluate_attack, evaluate_model, compute_confidence_interval
from statsmodels.stats.proportion import confint_proportions_2indep
import numpy as np
from utils import create_subset_dataloader
import torch
import os.path
import argparse

def main():
    parser = argparse.ArgumentParser(description='Evaluate model with adversarial attacks')
    parser.add_argument('--num_samples', type=int, default=100, 
                        help='Number of samples to use for evaluation (default: 100)')
    args = parser.parse_args()

    device = get_device()
    model = LittleCNN().to(device)

    if not os.path.exists("little_cnn.pth"):
        train_model(epochs=50, device=device)

    model.load_state_dict(torch.load('little_cnn.pth', map_location=device))
    _, test_loader = get_dataloaders()
    model.eval()

    num_samples = min(args.num_samples, len(test_loader.dataset))
    test_subset = create_subset_dataloader(test_loader, num_samples)

    PGD_metrics = evaluate_attack(model, test_subset, PGD, eps=0.05, device=device)
    DeepFool_metrics = evaluate_attack(model, test_subset, DeepFool, eps=0.05, device=device)

    df_success = np.array(DeepFool_metrics[1])
    PGD_success = np.array(PGD_metrics[1])
    count_df = df_success.sum()
    count_pgd = PGD_success.sum()
    nobs = len(df_success)

    ci = confint_proportions_2indep(
        count1=count_df, nobs1=nobs,
        count2=count_pgd, nobs2=nobs,
        method='agresti-caffo'
    )

    pgd_ci = compute_confidence_interval(PGD_metrics[1])
    print(f"\nPGD ASR: {PGD_metrics[0]:.2f}%")
    print(f"95% доверительный интервал для PGD ASR: [{pgd_ci[0]*100:.2f}%, {pgd_ci[1]*100:.2f}%]")

    DeepFool_ci = compute_confidence_interval(DeepFool_metrics[1])
    print(f"\nDeepFool ASR: {DeepFool_metrics[0]:.2f}%")
    print(f"95% доверительный интервал для PGD ASR: [{DeepFool_ci[0]*100:.2f}%, {DeepFool_ci[1]*100:.2f}%]")

    print("\n--- Сравнение успешности атак ---")
    print(f"\nDeepFool успешность: {count_df / nobs:.2%}")
    print(f"PGD успешность: {count_pgd / nobs:.2%}")
    print(f"Разница: {(count_df - count_pgd) / nobs:.2%}")
    print(f"95% Доверительный интервал: [{ci[0]:.2%}, {ci[1]:.2%}]")

    if ci[0] > 0:
        print("\nDeepFool значимо лучше PGD\n")
    elif ci[1] < 0:
        print("\nPGD значимо лучше DeepFool\n")
    else:
        print("\nМетоды одинаково эффективны\n")

if __name__ == "__main__":
    main()