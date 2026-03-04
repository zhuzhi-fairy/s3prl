# %% [markdown]
# # Audio Length Distribution
# Scan a dataset directory for audio files and plot the distribution of their durations.

# %% Configuration — edit these before running
from pathlib import Path

DATASET_DIR = Path("/mnt/data/zhu/Database/speech_commands/speech_commands_v0.01")
EXTENSIONS = {".wav", ".flac", ".ogg", ".mp3", ".opus", ".sph"}
BINS = 50

# %% Collect audio files
from tqdm import tqdm
import soundfile as sf

files = sorted(p for p in DATASET_DIR.rglob("*") if p.is_file() and p.suffix.lower() in EXTENSIONS)
print(f"Found {len(files)} audio files in {DATASET_DIR.name}")

# %% Read durations
durations = []
errors = []
for f in tqdm(files, desc="Reading durations", unit="file"):
    try:
        info = sf.info(str(f))
        durations.append(info.duration)
    except Exception as e:
        errors.append(f"{f}: {e}")

if errors:
    print(f"\nWarning: {len(errors)} files could not be read:")
    for e in errors[:10]:
        print(f"  {e}")

print(f"\nSuccessfully read {len(durations)} files")

# %% Summary statistics
import numpy as np

durations = np.array(durations)

print(f"Total files : {len(durations)}")
print(f"Total hours : {durations.sum() / 3600:.2f}")
print(f"Mean        : {durations.mean():.3f} s")
print(f"Std         : {durations.std():.3f} s")
print(f"Min         : {durations.min():.3f} s")
print(f"25th pct    : {np.percentile(durations, 25):.3f} s")
print(f"Median      : {np.median(durations):.3f} s")
print(f"75th pct    : {np.percentile(durations, 75):.3f} s")
print(f"Max         : {durations.max():.3f} s")

# %% Plot histogram
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(durations, bins=BINS, edgecolor="black", linewidth=0.5, alpha=0.8)
ax.set_xlabel("Duration (seconds)", fontsize=12)
ax.set_ylabel("Number of files", fontsize=12)
ax.set_title(f"Audio Length Distribution: {DATASET_DIR.name}", fontsize=14)

mean_val = durations.mean()
ax.axvline(mean_val, color="red", linestyle="--", linewidth=1.5, label=f"Mean = {mean_val:.3f} s")
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# %% Per-subfolder breakdown (optional)
from collections import Counter

subfolder_counts = Counter()
subfolder_durations: dict[str, list[float]] = {}

for f, d in zip(files, durations):
    rel = f.relative_to(DATASET_DIR)
    folder = rel.parts[0] if len(rel.parts) > 1 else "(root)"
    subfolder_counts[folder] += 1
    subfolder_durations.setdefault(folder, []).append(d)

print(f"{'Subfolder':<25} {'Count':>7} {'Mean (s)':>10} {'Total (min)':>12}")
print("-" * 58)
for folder in sorted(subfolder_counts):
    durs = np.array(subfolder_durations[folder])
    print(f"{folder:<25} {len(durs):>7} {durs.mean():>10.3f} {durs.sum() / 60:>12.1f}")

# %% Plot per-subfolder mean duration (optional)
folders_sorted = sorted(subfolder_durations.keys())
means = [np.mean(subfolder_durations[f]) for f in folders_sorted]

fig, ax = plt.subplots(figsize=(max(8, len(folders_sorted) * 0.4), 5))
ax.barh(folders_sorted, means, edgecolor="black", linewidth=0.5, alpha=0.8)
ax.set_xlabel("Mean Duration (seconds)", fontsize=12)
ax.set_title(f"Mean Duration per Subfolder: {DATASET_DIR.name}", fontsize=14)
ax.grid(True, axis="x", alpha=0.3)
plt.tight_layout()
plt.show()
