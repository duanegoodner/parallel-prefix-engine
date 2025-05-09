from pathlib import Path

path = Path("src/cuda_prefix_sum/CMakeLists.txt")
lines = path.read_text().splitlines()

new_lines = []
in_link_block = False
fixed = False

for line in lines:
    stripped = line.strip()
    
    # Fix the broken INTERFACE block
    if stripped.startswith("target_link_libraries(prefix_sum_cuda INTERFACE"):
        in_link_block = True
        new_lines.append(line)
        continue

    if in_link_block:
        if stripped.endswith(")"):
            in_link_block = False
            # Add the missing targets inside this INTERFACE block
            new_lines.append("    cuda_solver")
            new_lines.append("    kernel_launch_params")
            new_lines.append("    subtile_kernel")
            new_lines.append(")")
            fixed = True
        else:
            continue  # skip old misplaced lines
    else:
        new_lines.append(line)

# Add separate PRIVATE link block at end (if not already present)
if not fixed:
    new_lines.append("")
    new_lines.append("target_link_libraries(prefix_sum_cuda INTERFACE")
    new_lines.append("    cuda_solver")
    new_lines.append("    kernel_launch_params")
    new_lines.append("    subtile_kernel")
    new_lines.append(")")

new_lines.append("")
new_lines.append("target_link_libraries(prefix_sum_cuda PRIVATE")
new_lines.append("    CUDA::cudart")
new_lines.append("    cuda_headers")
new_lines.append("    my_default_flags")
new_lines.append(")")

# Write the corrected file
path.write_text("\n".join(new_lines))
print("✅ Fixed:", path)
