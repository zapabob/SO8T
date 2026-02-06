import torch
import math
from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl
import logging

# ロガー設定
logger = logging.getLogger(__name__)

class NKATDebugCallback(TrainerCallback):
    """
    NKAT SO(8) アダプターの状態をリアルタイム監視するコールバック
    """

    def __init__(self, model):
        self.model = model
        self.step_counter = 0

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        1ステップごとに呼び出される
        """
        self.step_counter += 1

        # ★★★ Alphaアニーリングを実装 ★★★
        # ステップごとにalphaを徐々に変化させる
        if hasattr(self.model, "named_modules"):
            total_steps = args.num_train_epochs * len(self.model.training_data) // args.per_device_train_batch_size if hasattr(self.model, 'training_data') else 1000
            progress = min(state.global_step / max(total_steps, 1), 1.0)

            for name, module in self.model.named_modules():
                if "nkat_adapter" in name and hasattr(module, "alpha_logit"):
                    # 負の値(-0.5)から正の値(0.5)へ線形アニーリング
                    target_alpha = -0.5 + progress * 1.0  # -0.5 to 0.5

                    # 逆sigmoid計算でlogitを更新
                    p = (target_alpha + 0.5) / 1.5
                    p = max(1e-7, min(1-1e-7, p))  # 数値安定性のためにclamp (float用)
                    new_logit = math.log(p / (1.0 - p))

                    # パラメータを直接更新（勾配なし）
                    with torch.no_grad():
                        module.alpha_logit.copy_(new_logit)

        # 10ステップに1回詳細ログを出す (頻度はお好みで調整)
        if self.step_counter % 5!= 0:
            return

        # モデルから SO(8) アダプターを探す
        adapters = []
        if hasattr(self.model, "named_modules"):
            for name, module in self.model.named_modules():
                if "nkat_adapter" in name and hasattr(module, "get_adapter_stats"):
                    adapters.append((name, module))

        if not adapters:
            return

        # ログ出力
        print(f"\n[NKAT DEBUG] Step {state.global_step} (Progress: {progress:.1%})")
        print(f"  Current Loss: {state.log_history[-1].get('loss', 'N/A') if state.log_history else 'N/A'}")

        for name, adapter in adapters[:3]: # 全部出すと多いので最初の3つだけ
            stats = adapter.get_adapter_stats()
            ortho_err = stats['orthogonality_error']
            alpha = stats['alpha']
            lie_norm = stats['lie_norm']

            # 勾配チェック
            grad_norm = "None"
            if adapter.lie_algebra.grad is not None:
                grad_norm = f"{torch.norm(adapter.lie_algebra.grad).item():.6f}"

            print(f"  - {name}:")
            print(f"    Ortho Error: {ortho_err:.6e}")
            print(f"    Alpha: {alpha:.6e}")
            print(f"    Lie Norm: {lie_norm:.6f}")
            print(f"    Grad Norm: {grad_norm}")

            # 異常検知
            if ortho_err > 1e-2:
                print("    [WARN] Orthogonality breaking down! (Check precision)")
            if alpha > 1.0:
                print("    [WARN] Alpha too large! (Risk of divergence)")
            if grad_norm == "None":
                print("    [WARN] Gradient detached! (Check requires_grad)")

        print("-" * 40)
