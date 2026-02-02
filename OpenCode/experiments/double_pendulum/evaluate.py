import torch
import matplotlib.pyplot as plt
import os
import sys
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.models.transformer_model import NKATTransformer, StandardTransformer

def evaluate():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    D_MODEL = 64
    HISTORY = 10

    # Load Models
    nkat = NKATTransformer(in_dim=HISTORY*4, d_model=D_MODEL, num_layers=2).to(device)
    base = StandardTransformer(in_dim=HISTORY*4, d_model=D_MODEL, num_layers=2).to(device)

    base_dir = os.path.dirname(__file__)
    try:
        nkat.load_state_dict(torch.load(os.path.join(base_dir, "nkat_pendulum.pth"), map_location=device))
        base.load_state_dict(torch.load(os.path.join(base_dir, "base_pendulum.pth"), map_location=device))
    except FileNotFoundError:
        print("Models not found. Run train.py first.")
        return

    data = torch.load(os.path.join(base_dir, "pendulum_data.pt"))
    SAMPLE_IDX = 300
    HISTORY = 10
    ROLLOUT_STEPS = 150

    gt_full = data[SAMPLE_IDX].numpy()
    init_hist = data[SAMPLE_IDX, :HISTORY, :].to(device)

    def run_rollout(model, hist, steps):
        model.eval()
        curr = hist.clone()
        preds = []
        with torch.no_grad():
            for _ in range(steps):
                inp = curr.flatten().unsqueeze(0)
                pred = model(inp)
                preds.append(pred.cpu().numpy()[0])
                curr = torch.cat([curr[1:], pred], dim=0)
        return np.array(preds)

    print(f"[PREDICT] Transformer Rollout Prediction...")
    pred_nkat = run_rollout(nkat, init_hist, ROLLOUT_STEPS)
    pred_base = run_rollout(base, init_hist, ROLLOUT_STEPS)

    time_axis = np.arange(ROLLOUT_STEPS)
    gt_future = gt_full[HISTORY : HISTORY+ROLLOUT_STEPS, 0]

    plt.figure(figsize=(12, 6))
    plt.plot(time_axis, gt_future, 'k-', alpha=0.3, linewidth=4, label="Ground Truth")
    plt.plot(time_axis, pred_base[:, 0], 'r--', linewidth=2, label="Standard Transformer")
    plt.plot(time_axis, pred_nkat[:, 0], 'b-', linewidth=2, label="NKAT Transformer")

    plt.title(f"Transformer Wars: Physics Structure vs Standard Architecture", fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(base_dir, "rollout_comparison.png"), dpi=150)
    print(f"[SAVE] Result saved to rollout_comparison.png")

if __name__ == "__main__":
    evaluate()












