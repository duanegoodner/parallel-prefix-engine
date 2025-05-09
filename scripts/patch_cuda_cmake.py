import os
import re

CUDA_DEPENDENCIES = ["CUDA::cudart", "cuda_headers", "my_default_flags"]

def patch_cmake(file_path):
    with open(file_path, "r") as f:
        lines = f.readlines()

    new_lines = []
    patched = False
    for i, line in enumerate(lines):
        if "target_link_libraries" in line:
            # Look for single-line usage
            match = re.match(r"(target_link_libraries\([^\)]+)", line.strip())
            if match:
                # If it's one-line, convert to multi-line
                target = match.group(1)
                new_lines.append(target + "\n")
                new_lines.append("    PRIVATE\n")
                for dep in CUDA_DEPENDENCIES:
                    new_lines.append(f"        {dep}\n")
                new_lines.append(")\n")
                patched = True
                continue
        elif line.strip() == ")" and not patched:
            # Look for closing ')' in multi-line style
            for dep in CUDA_DEPENDENCIES:
                new_lines.append(f"        {dep}\n")
            patched = True

        new_lines.append(line)

    if patched:
        with open(file_path, "w") as f:
            f.writelines(new_lines)
        print(f"[PATCHED] {file_path}")
    else:
        print(f"[SKIP] {file_path} — no patch needed or already correct")


def main():
    root_dir = "src/cuda_prefix_sum"
    for dirpath, _, filenames in os.walk(root_dir):
        for fname in filenames:
            if fname == "CMakeLists.txt":
                patch_cmake(os.path.join(dirpath, fname))


if __name__ == "__main__":
    main()
