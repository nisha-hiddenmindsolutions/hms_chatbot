import sys
import runpy
from pathlib import Path

# Add project root directory to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Run main Streamlit application script cleanly on every rerun
ui_script = PROJECT_ROOT / "app" / "ui" / "streamlit_app.py"
runpy.run_path(str(ui_script), run_name="__main__")
