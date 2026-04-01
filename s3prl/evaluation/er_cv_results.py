# %%
import json
import os

import numpy as np

exp_root = "result/downstream/er"
model_name = "24to24"
window_sizes = [128, 256, 512, 768]
lookahead_sizes = [0, 3, 15, 63]
result_message = "| context length | lookahead size = 0 | lookahead size = 3 | lookahead size = 15 | lookahead size = 63 |\n"
result_message += "| --- | --- | --- | --- | --- |\n"
for window_size in window_sizes:
    mess = f"| {window_size} "
    for lookahead_size in lookahead_sizes:
        exp_path = os.path.join(
            exp_root, f"{model_name}_ws{window_size}_la{lookahead_size}"
        )
        acc_ls = []
        for fold in range(1, 6):
            log_file = os.path.join(exp_path, f"fold{fold}", "log.log")
            with open(log_file, "r") as f:
                lines = f.readlines()
                acc_ls.append(float(lines[-1].strip().split()[-1]))
        cv_result = {
            "acc": acc_ls,
            "acc_mean": np.mean(acc_ls),
            "acc_std": np.std(acc_ls),
        }
        with open(os.path.join(exp_path, "cv_result.json"), "w") as f:
            json.dump(cv_result, f, indent=4)
        mess += f"| {cv_result['acc_mean']:.4f} "
    mess += "|\n"
    result_message += mess
print(result_message)
