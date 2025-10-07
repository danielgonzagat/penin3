# PENIN³ Package - Sistema de Inteligência REAL
import sys
from pathlib import Path

# Add paths to ensure imports work
PENIN3_ROOT = Path(__file__).parent
PROJECT_ROOT = PENIN3_ROOT.parent

# Add peninaocubo to path
sys.path.insert(0, str(PROJECT_ROOT / "peninaocubo"))

# Add intelligence_system to path  
sys.path.insert(0, str(PROJECT_ROOT / "intelligence_system"))

# Make penin3_system importable
sys.path.insert(0, str(PENIN3_ROOT))

__version__ = "3.0.0-real"
__all__ = ["PENIN3System", "PENIN3RealSystem"]
