from pathlib import Path
import runpy

ROOT = Path(__file__).resolve().parent
SCRIPT = ROOT / 'backend' / 'scripts' / 'clean_operational_data.py'

if __name__ == '__main__':
    runpy.run_path(str(SCRIPT), run_name='__main__')
