from __future__ import annotations
import csv
import hashlib
import os
import platform
import subprocess
import sys
from pathlib import Path
import numpy as np
import torch
import torchvision
from PIL import Image
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchvision import transforms as T
from torchvision.models import ResNet18_Weights

HERE = Path(__file__).resolve().parent

CLASSES = ["buildings", "forest", "glacier", "mountain", "sea", "street"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
IMG_SIZE = 160
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

N_TRAIN_PER_CLASS = 200
EPOCHS = 8
BATCH_TRAIN = 32
BATCH_EVAL = 64
LR_HEAD = 3e-3
LR_BACKBONE = 3e-4
WEIGHT_DECAY = 1e-4

TOTAL_PARAMS = 11_179_590
EXPECTED_TRAINABLE = {"scratch": 11_179_590, "frozen": 3_078, "unfreeze": 8_396_806}
TOP_MODULES = ["conv1", "bn1", "layer1", "layer2", "layer3", "layer4", "fc"]
EVAL_MODULES = {
    "scratch": [],
    "frozen": ["conv1", "bn1", "layer1", "layer2", "layer3", "layer4"],
    "unfreeze": ["conv1", "bn1", "layer1", "layer2", "layer3"],
}
TRAINABLE_MODULES = {"scratch": None, "frozen": ["fc"], "unfreeze": ["layer4", "fc"]}
REGIMES = ["scratch", "frozen", "unfreeze"]
RUNS = [("scratch", 1), ("frozen", 1), ("frozen", 2), ("frozen", 3),
        ("unfreeze", 1), ("unfreeze", 2), ("unfreeze", 3)]

EXPECTED_SPLIT = {
    "buildings": {"train": (200, 1996761), "validation": (218, 4837323), "test": (219, 4837559)},
    "forest": {"train": (200, 2032625), "validation": (237, 5276962), "test": (237, 5291326)},
    "glacier": {"train": (200, 2183923), "validation": (276, 6135652), "test": (277, 6157250)},
    "mountain": {"train": (200, 1950345), "validation": (262, 5789535), "test": (263, 5849765)},
    "sea": {"train": (200, 1987613), "validation": (255, 5655068), "test": (255, 5668011)},
    "street": {"train": (200, 1841817), "validation": (250, 5541845), "test": (251, 5547533)},
}
PARTS = ["train", "validation", "test"]


def _list_files(folder: Path) -> list[str]:
    return sorted(f for f in os.listdir(folder) if f.lower().endswith(".jpg"))


def build_split(root: Path) -> list[dict]:
    root = Path(root)
    rows = []
    rng = np.random.default_rng(0)
    for lab in CLASSES:
        files = _list_files(root / "seg_train" / "seg_train" / lab)
        idx = np.arange(len(files))
        rng.shuffle(idx)
        for f in sorted(files[i] for i in idx[:N_TRAIN_PER_CLASS]):
            rows.append({"file": f, "label": lab, "part": "train"})
    rng = np.random.default_rng(0)
    val_rows, test_rows = [], []
    for lab in CLASSES:
        files = _list_files(root / "seg_test" / "seg_test" / lab)
        n = len(files)
        idx = np.arange(n)
        rng.shuffle(idx)
        cut = n // 2
        val_rows += [{"file": f, "label": lab, "part": "validation"}
                     for f in sorted(files[i] for i in idx[:cut])]
        test_rows += [{"file": f, "label": lab, "part": "test"}
                      for f in sorted(files[i] for i in idx[cut:])]
    return rows + val_rows + test_rows


def write_split(rows: list[dict], path: Path) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["file", "label", "part"], lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def read_split(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def split_table(rows: list[dict]) -> dict:
    out = {}
    for lab in CLASSES:
        for part in PARTS:
            names = [r["file"] for r in rows if r["label"] == lab and r["part"] == part]
            out[(lab, part)] = (len(names), sum(int(Path(n).stem) for n in names))
    return out


def part_rows(rows: list[dict], part: str) -> list[dict]:
    return [r for r in rows if r["part"] == part]


def image_path(root: Path, row: dict) -> Path:
    folder = "seg_train" if row["part"] == "train" else "seg_test"
    return Path(root) / folder / folder / row["label"] / row["file"]


class SplitDataset(Dataset):
    def __init__(self, rows, root, transform):
        self.rows, self.root, self.transform = rows, Path(root), transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        with Image.open(image_path(self.root, r)) as im:
            img = im.convert("RGB")
        return self.transform(img), CLASS_TO_IDX[r["label"]]


def train_transform():
    return T.Compose([T.RandomResizedCrop(IMG_SIZE, scale=(0.7, 1.0)), T.RandomHorizontalFlip(),
                      T.ToTensor(), T.Normalize(MEAN, STD)])


def eval_transform():
    return T.Compose([T.Resize((IMG_SIZE, IMG_SIZE)), T.ToTensor(), T.Normalize(MEAN, STD)])


def make_train_loader(rows, root):
    ds = SplitDataset(part_rows(rows, "train"), root, train_transform())
    return DataLoader(ds, batch_size=BATCH_TRAIN, shuffle=True, drop_last=False, num_workers=0)


def make_eval_loader(rows, root, part, cache=False):
    ds = SplitDataset(part_rows(rows, part), root, eval_transform())
    loader = DataLoader(ds, batch_size=BATCH_EVAL, shuffle=False, num_workers=0)
    if not cache:
        return loader
    xs, ys = zip(*[(x, y) for x, y in loader])
    return DataLoader(TensorDataset(torch.cat(xs), torch.cat(ys)), batch_size=BATCH_EVAL,
                      shuffle=False, num_workers=0)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def device_name(device: torch.device) -> str:
    if device.type == "cuda":
        return f"cuda: {torch.cuda.get_device_name(0)}"
    return device.type


def param_groups(model: nn.Module, regime: str) -> list[dict]:
    if regime == "scratch":
        return [{"name": "all", "params": list(model.parameters()), "lr": LR_HEAD}]
    groups = [{"name": "fc", "params": list(model.fc.parameters()), "lr": LR_HEAD}]
    if regime == "unfreeze":
        groups.append({"name": "layer4", "params": list(model.layer4.parameters()), "lr": LR_BACKBONE})
    return groups


def build_model(regime: str, seed: int, device: torch.device):
    torch.manual_seed(seed)
    np.random.seed(seed)
    weights = None if regime == "scratch" else ResNet18_Weights.IMAGENET1K_V1
    model = torchvision.models.resnet18(weights=weights)
    model.fc = nn.Linear(512, 6)
    trainable = TRAINABLE_MODULES[regime]
    for p in model.parameters():
        p.requires_grad = trainable is None
    if trainable is not None:
        for name in trainable:
            for p in getattr(model, name).parameters():
                p.requires_grad = True
    model.to(device)
    optimizer = torch.optim.AdamW(param_groups(model, regime), weight_decay=WEIGHT_DECAY)
    return model, optimizer


def set_train_mode(model: nn.Module, regime: str) -> None:
    model.train()
    for name in EVAL_MODULES[regime]:
        getattr(model, name).eval()


def count_params(params) -> int:
    return sum(p.numel() for p in params)


@torch.no_grad()
def predict_logits(model: nn.Module, loader: DataLoader, device: torch.device) -> np.ndarray:
    model.eval()
    out = [model(x.to(device)).float().cpu() for x, _ in loader]
    return torch.cat(out).numpy()


def macro_f1(y_true, logits) -> float:
    return float(f1_score(y_true, np.asarray(logits).argmax(1), average="macro",
                          labels=list(range(6)), zero_division=0))


def labels_of(rows: list[dict], part: str) -> np.ndarray:
    return np.array([CLASS_TO_IDX[r["label"]] for r in part_rows(rows, part)])


def select_run(final_f1: dict) -> tuple[str, int]:
    means = {}
    for r in REGIMES:
        vals = [f for (rr, _), f in final_f1.items() if rr == r]
        if vals:
            means[r] = sum(vals) / len(vals)
    best_regime = max(means, key=lambda r: means[r])
    seeds = [s for (rr, s) in final_f1 if rr == best_regime]
    best_seed = max(seeds, key=lambda s: (final_f1[(best_regime, s)], -s))
    return best_regime, best_seed


def interpret(delta: float, R: float) -> str:
    if abs(delta) <= R:
        return "|Δ| ≤ R: різниця не відрізняється від розкиду - результату немає"
    return "|Δ| > R: ефект вартий перевірки, але не доведений"


def git_info() -> dict:
    try:
        h = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=HERE, text=True,
                                    stderr=subprocess.DEVNULL).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=HERE,
                                    text=True, stderr=subprocess.DEVNULL).strip())
        return {"commit": h, "uncommitted_changes": dirty}
    except Exception:
        return {"commit": None, "uncommitted_changes": None}


def versions() -> dict:
    return {"python": sys.version.split()[0], "torch": torch.__version__,
            "torchvision": torchvision.__version__, "platform": platform.platform()}
