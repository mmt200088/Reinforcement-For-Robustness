import torch
import matplotlib.pyplot as plt
from pathlib import Path

def approx_exp(x: torch.Tensor, degree: int) -> torch.Tensor:
    n = 2 ** degree
    return torch.pow(1 + x / n, n)

def main():
    # 只关注 x < 0
    x_min, x_max = -10.0, 0.0
    x = torch.linspace(x_min, x_max, steps=2000, dtype=torch.float64)

    y_exact = torch.exp(x)

    plt.figure()
    plt.plot(x.numpy(), y_exact.numpy(), label="exp(x) (exact)")

    for d in range(1, 7):
        y = approx_exp(x, d)
        plt.plot(x.numpy(), y.numpy(), label=f"degree={d} (n={2**d})")

    plt.xlabel("x (x<0)")
    plt.ylabel("y")
    plt.title("Approximation curves: (1 + x/2^d)^(2^d) vs exp(x)")
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.legend()

    # 保存到文件（改成你想要的路径）
    out_path = Path("approx_exp_curves_deg1-6.png")
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()  # 关闭图，避免占用内存/弹窗

    print(f"Saved to: {out_path.resolve()}")

if __name__ == "__main__":
    main()
