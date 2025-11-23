import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import os
import sys
from tqdm import tqdm

# Import Logic
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
# Import the new Transformer models
from src.models.transformer_model import NKATTransformer, StandardTransformer

def train():
    # --- Optimization Setup ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.backends.cudnn.benchmark = True
        print(f"[GPU] Powered by RTX 3060 ({torch.cuda.get_device_name(0)})")

    # 1. Load Data
    data_path = os.path.join(os.path.dirname(__file__), "pendulum_data.pt")
    if not os.path.exists(data_path):
        print("Error: pendulum_data.pt not found. Run generate_data.py first.")
        return

    print(f"📦 Loading data...")
    data = torch.load(data_path)
    N, S, D = data.shape

    HISTORY = 10
    inputs, targets = [], []
    TRAIN_STEPS = 150

    print("⚙️  Processing dataset...")
    for i in tqdm(range(N), desc="Building Tensors"):
        for t in range(HISTORY, TRAIN_STEPS):
            hist = data[i, t-HISTORY:t, :].flatten()
            tgt = data[i, t, :]
            inputs.append(hist)
            targets.append(tgt)

    inputs = torch.stack(inputs)
    targets = torch.stack(targets)

    dataset = TensorDataset(inputs, targets)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])

    # Tuned for RTX 3060 + Ryzen
    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True, num_workers=0, pin_memory=True)

    # 2. Setup Models (Transformer Wars)
    D_MODEL = 64
    # Standard Transformer
    model_base = StandardTransformer(in_dim=HISTORY*4, d_model=D_MODEL, num_layers=2).to(device)
    # NKAT Transformer
    model_nkat = NKATTransformer(in_dim=HISTORY*4, d_model=D_MODEL, num_layers=2).to(device)

    print(f"[BATTLE] Model Comparison: Standard Transformer vs NKAT Transformer")

    opt_nkat = optim.Adam(model_nkat.parameters(), lr=5e-4) # Lower LR for Transformers
    opt_base = optim.Adam(model_base.parameters(), lr=5e-4)
    criterion = nn.MSELoss()

    # 3. Training Loop - TRANSFORMER WARS
    EPOCHS = 30
    hist_nkat, hist_base = [], []

    print(f"[BATTLE] TRANSFORMER WARS: {EPOCHS} rounds commencing!")
    print(f"[ARENA] RTX 3060 Double Pendulum Chaos Prediction")
    print("=" * 60)

    # Overall battle progress
    battle_pbar = tqdm(range(EPOCHS), desc="Battle Progress", position=0, leave=True)

    for ep in range(EPOCHS):
        model_nkat.train(); model_base.train()
        loss_n_sum, loss_b_sum = 0, 0
        pbar = tqdm(train_loader, desc=f"Epoch {ep+1}/{EPOCHS}", leave=False)

        for x, y in pbar:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

            # NKAT
            opt_nkat.zero_grad()
            loss_n = criterion(model_nkat(x), y)
            loss_n.backward()
            opt_nkat.step()

            # Baseline
            opt_base.zero_grad()
            loss_b = criterion(model_base(x), y)
            loss_b.backward()
            opt_base.step()

            loss_n_sum += loss_n.item()
            loss_b_sum += loss_b.item()

            # Battle status display
            nkat_loss = loss_n.item()
            base_loss = loss_b.item()
            winner = "NKAT" if nkat_loss < base_loss else "Standard" if base_loss < nkat_loss else "TIE"
            advantage = abs(nkat_loss - base_loss)

            pbar.set_postfix({
                "NKAT": f"{nkat_loss:.4f}",
                "STD": f"{base_loss:.4f}",
                "Lead": winner[:4],
                "Gap": f"{advantage:.4f}"
            })

        avg_n = loss_n_sum / len(train_loader)
        avg_b = loss_b_sum / len(train_loader)
        hist_nkat.append(avg_n)
        hist_base.append(avg_b)

        # Battle result display
        round_winner = "NKAT WINS" if avg_n < avg_b else "Standard WINS" if avg_b < avg_n else "DRAW"
        performance_diff = abs(avg_n - avg_b)

        print(f"  [BATTLE {ep+1:2d}] NKAT={avg_n:.6f} | Standard={avg_b:.6f} | {round_winner} (diff: {performance_diff:.6f})")

        # Update overall battle progress
        battle_pbar.set_postfix({
            "Current Round": f"{ep+1}/{EPOCHS}",
            "Latest Winner": round_winner.split()[0],
            "NKAT_Avg": f"{avg_n:.4f}",
            "Standard_Avg": f"{avg_b:.4f}"
        })

    battle_pbar.close()

    torch.save(model_nkat.state_dict(), os.path.join(os.path.dirname(__file__), "nkat_pendulum.pth"))
    torch.save(model_base.state_dict(), os.path.join(os.path.dirname(__file__), "base_pendulum.pth"))

    # Final battle summary
    nkat_wins = sum(1 for n, b in zip(hist_nkat, hist_base) if n < b)
    base_wins = sum(1 for n, b in zip(hist_nkat, hist_base) if b < n)
    draws = EPOCHS - nkat_wins - base_wins

    final_nkat = hist_nkat[-1]
    final_base = hist_base[-1]

    print("\n" + "="*60)
    print("🏆 TRANSFORMER WARS FINAL RESULTS 🏆")
    print("="*60)
    print(f"Total Rounds: {EPOCHS}")
    print(f"NKAT Wins: {nkat_wins}")
    print(f"Standard Wins: {base_wins}")
    print(f"Draws: {draws}")
    print(f"")
    print(f"Final Scores:")
    print(f"NKAT Transformer: {final_nkat:.6f}")
    print(f"Standard Transformer: {final_base:.6f}")
    print(f"")
    if final_nkat < final_base:
        print("🏆 WINNER: NKAT TRANSFORMER (Physics Structure Victory!) 🏆")
        print(f"   Advantage: {final_base - final_nkat:.6f}")
    elif final_base < final_nkat:
        print("🏆 WINNER: Standard TRANSFORMER (Traditional Architecture Victory!) 🏆")
        print(f"   Advantage: {final_nkat - final_base:.6f}")
    else:
        print("🤝 DRAW: Both transformers equally matched!")
    print("="*60)
    print("[COMPLETE] Training complete.")


    plt.figure(figsize=(10, 6))
    plt.plot(hist_nkat, label="NKAT Transformer", linewidth=2.5, color='blue')
    plt.plot(hist_base, label="Standard Transformer", linestyle="--", color='red', alpha=0.7)
    plt.title(f"Learning Curve: Transformer Comparison", fontsize=14)
    plt.xlabel("Epochs")
    plt.ylabel("MSE Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(os.path.dirname(__file__), "train_loss.png"))
    print("[SAVE] Saved train_loss.png")

if __name__ == "__main__":
    train()

    print("[SAVE] Saved train_loss.png")

if __name__ == "__main__":
    train()

    print("[SAVE] Saved train_loss.png")

if __name__ == "__main__":
    train()