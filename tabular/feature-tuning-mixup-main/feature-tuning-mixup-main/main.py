import argparse
from datetime import datetime
import torch
import numpy as np
import random

from Data_loader import load_data
from preprocessing import preprocess
from models import MLP_Surrogate, MLP_Target
from train_models import train_mlp, train_sklearn_models
from ftm_attack import fgsm_attack
from evaluate import evaluate_transfer

from sklearn.model_selection import train_test_split

now = datetime.now()
today_string = now.strftime("%Y-%m-%d|%H:%M:%S")


def main(args):
    print(today_string)
    print(args)
    print("#" * 50)

    # =========================
    # 🔁 SEED
    # =========================
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # =========================
    # 📊 LOAD DATA
    # =========================
    print("Loading dataset...")
    df = load_data()

    print("Preprocessing...")
    X, y = preprocess(df)

    # =========================
    # 🔀 TRAIN-TEST SPLIT
    # =========================
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed
    )

    input_dim = X.shape[1]

    # =========================
    # 🧠 TRAIN MODELS
    # =========================
    print("Training models...")

    surrogate = train_mlp(MLP_Surrogate(input_dim), X_train, y_train)
    target_mlp = train_mlp(MLP_Target(input_dim), X_train, y_train)

    lr, rf = train_sklearn_models(X_train, y_train)

    print("Models trained.")

    # =========================
    # 🎯 SELECT ATTACK SAMPLES
    # =========================
    print("Selecting attack samples...")

    X_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_tensor = torch.tensor(y_test.values, dtype=torch.long)

    # Only correctly classified samples
    with torch.no_grad():
        preds = surrogate(X_tensor).argmax(dim=1)

    correct_mask = (preds.numpy() == y_test.to_numpy())

    # Attack only class 0 → 1
    final_mask = (y_test.to_numpy() == 0) & correct_mask

    x_attack = X_tensor[final_mask]
    y_attack = y_tensor[final_mask]

    # Target labels = 1
    y_target = torch.ones(len(x_attack), dtype=torch.long)

    print(f"Number of attack samples: {len(x_attack)}")

    # =========================
    # ⚔️ ATTACK
    # =========================
    print("Generating adversarial examples...")

    x_adv = fgsm_attack(
        surrogate,
        x_attack,
        y_target,
        epsilon=args.epsilon
    )

    # =========================
    # 📊 EVALUATION
    # =========================
    print("Evaluating transferability...")

    models = {
        "surrogate": surrogate,
        "mlp_target": target_mlp,
        "logistic_regression": lr,
        "random_forest": rf
    }

    results = evaluate_transfer(
        models,
        x_adv.numpy(),
        y_target.numpy()
    )

    print("\n========== RESULTS ==========")
    for name, score in results.items():
        print(f"{name}: {score:.4f}")

    print("DONE")


# =========================
# ⚙️ ARGUMENTS
# =========================
def argument_parsing():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.1,
        help="perturbation strength"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="random seed"
    )

    return parser


if __name__ == "__main__":
    args = argument_parsing().parse_args()
    main(args)