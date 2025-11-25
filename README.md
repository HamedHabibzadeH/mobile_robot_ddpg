# Mobile Robot Path Finding with DDPG (Gymnasium + SB3)

این ریپو شامل کد کامل محیط ناوبری ربات سیار با LiDAR و آموزش با DDPG است.

## نصب سریع

> Python 3.9–3.11 پیشنهاد می‌شود.

```bash
# (اختیاری) ساخت venv
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell
# .venv\Scripts\Activate.ps1

# به‌روزرسانی ابزارهای نصب
python -m pip install --upgrade pip setuptools wheel

# نصب پیش‌نیازها
pip install -r requirements.txt

# نصب PyTorch (بر اساس سیستم خودتان)
# فقط CPU:
pip install torch --index-url https://download.pytorch.org/whl/cpu
# یا طبق CUDA: https://pytorch.org/get-started/locally/
```

## آموزش

```bash
python train_ddpg.py --logdir runs_ddpg --n_envs 1
```

## اجرای دمو / ارزیابی

```bash
python evaluate.py --model runs_ddpg/best_model.zip --episodes 5 --deterministic --sleep 0.02
```

## نکات

- اگر روی سرور بدون نمایشگر اجرا می‌کنید، `render()` ممکن است باز نشود. می‌توانید `--sleep 0` بزنید و از لاگ‌ها استفاده کنید، یا از بک‌اند غیرتعامل `Agg` استفاده کنید:
  ```bash
  MPLBACKEND=Agg python evaluate.py --model runs_ddpg/best_model.zip
  ```
- برای سرعت بیشتر، `LIDAR_N_RAYS` یا `TOTAL_TIMESTEPS` را کم کنید.
