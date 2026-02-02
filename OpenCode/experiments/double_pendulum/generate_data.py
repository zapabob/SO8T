import numpy as np
import torch
from scipy.integrate import odeint
import os
from tqdm import tqdm

# --- Physics Params ---

L1, L2 = 1.0, 1.0
M1, M2 = 1.0, 1.0
G = 9.81
DT = 0.05
SEQ_LEN = 200
NUM_SAMPLES = 500  # Fast generation for transformer experiments

def derivs(state, t):
    dydx = np.zeros_like(state)
    th1, w1, th2, w2 = state

    delta = th2 - th1
    den1 = (M1 + M2) * L1 - M2 * L1 * np.cos(delta) * np.cos(delta)
    den2 = (L2 / L1) * den1

    dydx[0] = w1
    dydx[1] = ((M2 * L1 * w1 * w1 * np.sin(delta) * np.cos(delta) +
                M2 * G * np.sin(th2) * np.cos(delta) +
                M2 * L2 * w2 * w2 * np.sin(delta) -
                (M1 + M2) * G * np.sin(th1)) / den1)
    dydx[2] = w2
    dydx[3] = ((- M2 * L2 * w2 * w2 * np.sin(delta) * np.cos(delta) +
                (M1 + M2) * G * np.sin(th1) * np.cos(delta) -
                (M1 + M2) * L1 * w1 * w1 * np.sin(delta) -
                (M1 + M2) * G * np.sin(th2)) / den2)

    return dydx

def pol2cart(th1, th2):
    x1 = L1 * np.sin(th1)
    y1 = -L1 * np.cos(th1)
    x2 = x1 + L2 * np.sin(th2)
    y2 = y1 - L2 * np.cos(th2)
    return np.stack([x1, y1, x2, y2], axis=-1)

def generate_dataset():
    print(f"Generating {NUM_SAMPLES} episodes of Double Pendulum...")
    all_data = []
    t = np.arange(0, SEQ_LEN * DT, DT)

    for _ in tqdm(range(NUM_SAMPLES), desc="Simulating Physics"):
        th1 = np.random.uniform(-np.pi, np.pi)
        th2 = np.random.uniform(-np.pi, np.pi)
        w1 = np.random.uniform(-1, 1)
        w2 = np.random.uniform(-1, 1)
        state0 = [th1, w1, th2, w2]
        states = odeint(derivs, state0, t)
        coords = pol2cart(states[:, 0], states[:, 2])
        all_data.append(coords)

    dataset = np.array(all_data, dtype=np.float32)
    save_path = os.path.join(os.path.dirname(__file__), "pendulum_data.pt")
    torch.save(torch.from_numpy(dataset), save_path)

    print(f"Saved to {save_path}")
    print(f"Data shape: {dataset.shape}")

if __name__ == "__main__":
    generate_dataset()
