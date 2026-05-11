import os
import zipfile
from pathlib import Path

def archive_project_code():
    # Configuration
    source_dirs = ['dataset_builder', 'frontend', 'backend', 'protonet']
    output_filename = 'ReviewOp_Flattened_Source.zip'
    
    # Exclude patterns
    exclude_dirs = {
        'venv', '.venv', '__pycache__', 'node_modules', '.git', 
        '.idea', '.vscode', 'dist', 'build', 'cache', '.pytest_cache','.tmp'
    }
    
    # Allowed extensions (Important codes)
    allowed_extensions = {
        '.py', '.js', '.jsx', '.ts', '.tsx', '.css', '.html', 
        '.json', '.md', '.sql', '.yaml', '.yml', '.toml', '.ps1', '.sh', '.zip'
    }
    
    # Files to explicitly ignore even if they have allowed extensions
    ignore_files = {
        'package-lock.json', 'yarn.lock', '.env', '.env.local', 
        '.DS_Store', 'ReviewOp_Flattened_Source.zip'
    }

    repo_root = Path.cwd()
    
    print(f"Starting archival into {output_filename}...")
    count = 0
    
    with zipfile.ZipFile(output_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
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
                    
                    # Add to zip
                    zipf.write(file_path, flattened_name)
                    count += 1
                    print(f"Added: {flattened_name}")

    print(f"\nSuccessfully archived {count} files into {output_filename}")

if __name__ == "__main__":
    archive_project_code()
