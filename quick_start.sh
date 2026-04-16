#!/usr/bin/env python
"""Quick start script for the project."""
import subprocess
import sys
import os
from pathlib import Path

def run_command(cmd, description):
    """Run a shell command with error handling."""
    print(f"\n📌 {description}...")
    result = subprocess.run(cmd, shell=True, cwd=Path(__file__).parent)
    if result.returncode != 0:
        print(f"❌ Failed: {description}")
        return False
    print(f"✅ {description}")
    return True


def main():
    """Run quick setup."""
    print("=" * 60)
    print("StyleSync - MLOps Project Setup")
    print("=" * 60)
    
    # Check Python version
    if sys.version_info < (3, 9):
        print("❌ Python 3.9+ required")
        sys.exit(1)
    
    print("✅ Python version OK")
    
    # Create virtual environment
    if not Path("venv").exists():
        run_command("python3 -m venv venv", "Creating virtual environment")
    
    # Activate and install
    activate_cmd = "source venv/bin/activate" if os.name != "nt" else "venv\\Scripts\\activate"
    
    run_command(f"{activate_cmd} && pip install -r requirements.txt", "Installing dependencies")
    
    # Setup project
    run_command("python setup.py", "Creating project structure")
    
    print("\n" + "=" * 60)
    print("Setup Complete! 🎉")
    print("=" * 60)
    print("\nTo start development:")
    print("  1. Activate venv: source venv/bin/activate")
    print("  2. Start services: docker-compose up -d")
    print("  3. Run backend: python -m backend.main")
    print("  4. Run frontend (new terminal): streamlit run frontend/app.py")


if __name__ == "__main__":
    main()
