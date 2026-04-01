# %%
import os

exp_root = "result/downstream/sd"
model_name = "24to24"
window_sizes = [128, 256, 512, 768]
lookahead_sizes = [0, 3, 15, 63]
result_message = "| context length | lookahead size = 0 | lookahead size = 3 | lookahead size = 15 | lookahead size = 63 |\n"
result_message += "| --- | --- | --- | --- | --- |\n"
for window_size in window_sizes:
    mess = f"| {window_size} "
    for lookahead_size in lookahead_sizes:
        log_file = os.path.join(
            exp_root, f"{model_name}_ws{window_size}_la{lookahead_size}/evaluation.log"
        )
        with open(log_file, "r") as f:
            lines = f.readlines()
            acc = float(lines[-1].strip().split()[3])
            der = float(lines[-1].strip().split()[5])
        mess += f"| acc={acc:.04f}, der={der:.04f} "
    mess += "|\n"
    result_message += mess
print(result_message)

# %%
