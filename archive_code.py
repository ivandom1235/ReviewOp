import os
import zipfile
from pathlib import Path

def archive_project_code():
    # Configuration
    source_dirs = ['dataset_builder', 'protonet', 'backend', 'frontend']
    output_filename = 'ReviewOp_Flattened_Source.zip'
    output_tree_filename = 'ReviewOp_Reproducible_Source.zip'
    reproducibility_root_files = (
        "requirements.txt",
        "run_wsl.sh",
        "run_repro_strict.sh",
        "CURRENT_ACTIVE_ARTIFACT.json",
    )
    
    # Exclude patterns
    exclude_dirs = {
        'venv', '.venv', '__pycache__', 'node_modules', '.git', 
        '.idea', '.vscode', 'dist', 'build', 'cache', '.pytest_cache','.tmp'
    }
    
    # Allowed extensions (Important codes)
    allowed_extensions = {
        '.py', '.js', '.jsx', '.ts', '.tsx', '.css', '.html', 
        '.json', '.jsonl', '.md', '.sql', '.yaml', '.yml', '.toml', '.ps1', '.sh', '.zip', '.txt', '.lock'
    }
    
    # Files to explicitly ignore even if they have allowed extensions
    ignore_files = {
        'yarn.lock', '.env', '.env.local', 
        '.DS_Store', 'ReviewOp_Flattened_Source.zip', '.llm_cache.db'
    }

    repo_root = Path.cwd()
    
    print(f"Starting archival into {output_filename}...")
    count = 0
    used_arc_names = set()

    def unique_arcname(name: str) -> str:
        """Return a unique archive entry name while preserving readability."""
        if name not in used_arc_names:
            used_arc_names.add(name)
            return name
        stem, dot, suffix = name.rpartition(".")
        if not dot:
            stem, suffix = name, ""
        i = 2
        while True:
            candidate = f"{stem}__dup{i}"
            if suffix:
                candidate = f"{candidate}.{suffix}"
            if candidate not in used_arc_names:
                used_arc_names.add(candidate)
                return candidate
            i += 1
    
    with zipfile.ZipFile(output_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # Include minimal root reproducibility files.
        for root_file in reproducibility_root_files:
            rf = repo_root / root_file
            if rf.exists() and rf.is_file():
                arcname = unique_arcname(root_file)
                zipf.write(rf, arcname)
                count += 1
                print(f"Added root reproducibility file: {arcname}")

        # Archive source directories
        for s_dir in source_dirs:
            target_path = repo_root / s_dir
            if not target_path.exists():
                print(f"Warning: Directory {s_dir} not found. Skipping.")
                continue
                
            for root, dirs, files in os.walk(target_path):
                # Filter out excluded directories in-place
                dirs[:] = [d for d in dirs if d not in exclude_dirs]
                
                for file in files:
                    if file in ignore_files:
                        continue
                        
                    file_path = Path(root) / file
                    
                    # Check extension
                    if file_path.suffix.lower() not in allowed_extensions:
                        continue
                        
                    # Get relative path from repo root
                    try:
                        rel_path = file_path.relative_to(repo_root)
                    except ValueError:
                        # Fallback if somehow not under repo_root
                        rel_path = Path(s_dir) / file_path.name
                    
                    # Flatten filename: replace path separators with underscores
                    flattened_name = str(rel_path).replace(os.sep, '_')
                    flattened_name = unique_arcname(flattened_name)
                    
                    # Add to zip
                    zipf.write(file_path, flattened_name)
                    count += 1
                    print(f"Added: {flattened_name}")

    print(f"\nSuccessfully archived {count} files into {output_filename}")

    # Reproducible archive preserving folder structure
    tree_count = 0
    with zipfile.ZipFile(output_tree_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root_file in reproducibility_root_files:
            rf = repo_root / root_file
            if rf.exists() and rf.is_file():
                zipf.write(rf, root_file)
                tree_count += 1

        for s_dir in source_dirs:
            target_path = repo_root / s_dir
            if not target_path.exists():
                continue
            for root, dirs, files in os.walk(target_path):
                dirs[:] = [d for d in dirs if d not in exclude_dirs]
                for file in files:
                    if file in ignore_files:
                        continue
                    file_path = Path(root) / file
                    if file_path.suffix.lower() not in allowed_extensions:
                        continue
                    rel_path = file_path.relative_to(repo_root)
                    zipf.write(file_path, str(rel_path))
                    tree_count += 1
    print(f"Successfully archived {tree_count} files into {output_tree_filename}")

if __name__ == "__main__":
    archive_project_code()
