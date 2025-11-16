from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
LOSS_CURVE_PATH = SCRIPT_DIR / "loss_curve.png"
MODEL_PATH = SCRIPT_DIR / "model.pth"
BATCH_SIZE = 64
EPOCHS = 20