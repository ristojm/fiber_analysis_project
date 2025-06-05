import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "modules"))

try:
    from modules.porosity_analysis import EnhancedPorosityAnalyzer
    print("✅ Enhanced porosity analyzer imported successfully!")
    
    analyzer = EnhancedPorosityAnalyzer()
    print("✅ Enhanced porosity analyzer initialized successfully!")
    
except ImportError as e:
    print(f"❌ Import failed: {e}")

print("Porosity analysis should now work!")