import os
import tempfile
import tqdm
import torch
from torch.utils.data.dataloader import DataLoader

from src.flow_match_dit.flowMatch import FlowMatch


def safe_save(obj, path):
    """原子写盘,防止半截文件。"""
    dir_ = os.path.dirname(os.path.abspath(path))
    os.makedirs(dir_, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=dir_, suffix=".tmp")
    os.close(fd)
    try:
        torch.save(obj, tmp)
        os.replace(tmp, path)
    except Exception:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def rotate_ckpts(dir_, prefix="ckpt_step", keep=3):
    files = sorted(
        (f for f in os.listdir(dir_)
         if f.startswith(prefix) and f.endswith(".pt")),
        key=lambda f: int(f[len(prefix):-3]),
    )
    for f in files[:-keep]:
        os.remove(os.path.join(dir_, f))


def train(
    model: FlowMatch,
    optimizer: torch.optim.Optimizer,
    epochs: int,
    device: str,
    train_loader: DataLoader,
    val_loader: DataLoader | None = None,
    save_dir: str = "checkpoints",
    save_interval: int = 500,
    keep_last: int = 3,
    grad_clip: float = 1.0,
    log_every: int = 50,
    start_step: int = 0,
):
    os.makedirs(save_dir, exist_ok=True)
    train_losses, val_losses = [], []
    step = start_step

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        n_batches = 0

        pbar = tqdm.tqdm(train_loader, desc=f"epoch {epoch}")
        for img, label in pbar:                       # ← 用 pbar 迭代, 进度条才会动
            img, label = img.to(device, non_blocking=True), label.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            loss = model(img, label)
            loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            running_loss += loss.item()
            n_batches += 1
            step += 1

            if step % log_every == 0:
                pbar.set_description(
                    f"epoch {epoch} | step {step} | loss {running_loss / n_batches:.4f}"
                )

            if save_interval > 0 and step % save_interval == 0:
                path = os.path.join(save_dir, f"ckpt_step{step:08d}.pt")
                safe_save({
                    "model":     model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "step":      step,
                    "epoch":     epoch,
                }, path)
                rotate_ckpts(save_dir, keep=keep_last)

        train_losses.append(running_loss / max(1, n_batches))

        if val_loader is not None:
            model.eval()
            v = 0.0
            nv = 0
            with torch.no_grad():
                for imgs, labels in val_loader:
                    imgs, labels = imgs.to(device), labels.to(device)
                    v += model(imgs, labels).item()
                    nv += 1
            val_losses.append(v / max(1, nv))
            print(f"[epoch {epoch}] train {train_losses[-1]:.4f} | val {val_losses[-1]:.4f}")
        else:
            print(f"[epoch {epoch}] train {train_losses[-1]:.4f}")

    # 训练结束存一份最终版
    safe_save({
        "model":     model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step":      step,
        "epoch":     epochs,
    }, os.path.join(save_dir, "ckpt_final.pt"))

    return train_losses, val_losses