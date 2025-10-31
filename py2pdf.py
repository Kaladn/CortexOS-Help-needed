import os

def list_files_recursively(root_dir, max_depth=3):
    file_list = []
    base_depth = root_dir.rstrip(os.path.sep).count(os.path.sep)
    for foldername, subfolders, filenames in os.walk(root_dir):
        current_depth = foldername.rstrip(os.path.sep).count(os.path.sep)
        if current_depth - base_depth >= max_depth:
            subfolders[:] = []  # Stop further descent
        file_list.append(f"Folder: {foldername}")
        for subfolder in subfolders:
            file_list.append(f"  [D] {subfolder}")
        for filename in filenames:
            file_list.append(f"  [F] {filename}")
        file_list.append("\n")
    return file_list
