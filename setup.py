#!/usr/bin/env python3
"""
Setup script for Brain Tumor MRI Classification project
"""

import os
import sys
import subprocess
import json
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 7):
        print("❌ Python 3.7 or higher is required")
        sys.exit(1)
    print(f"✅ Python {sys.version.split()[0]} detected")

def install_requirements():
    """Install required packages"""
    print("📦 Installing requirements...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Requirements installed successfully")
    except subprocess.CalledProcessError:
        print("❌ Failed to install requirements")
        sys.exit(1)

def setup_kaggle():
    """Setup Kaggle API configuration"""
    kaggle_dir = Path.home() / ".kaggle"
    kaggle_file = kaggle_dir / "kaggle.json"
    
    if not kaggle_file.exists():
        print("⚠️  Kaggle API not configured")
        print("📝 Please follow these steps:")
        print("1. Go to https://www.kaggle.com/account")
        print("2. Click 'Create New API Token'")
        print("3. Download kaggle.json")
        print(f"4. Place it in: {kaggle_file}")
        return False
    
    # Set proper permissions on Unix-like systems
    if os.name != 'nt':  # Not Windows
        os.chmod(kaggle_file, 0o600)
    
    print("✅ Kaggle API configured")
    return True

def create_directories():
    """Create necessary directories"""
    directories = [
        "models",
        "data",
        "results",
        "logs",
        "figures"
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
    
    print("✅ Project directories created")

def main():
    """Main setup function"""
    print("🧠 Brain Tumor MRI Classification - Setup")
    print("=" * 50)
    
    check_python_version()
    install_requirements()
    create_directories()
    
    kaggle_ready = setup_kaggle()
    
    print("\n" + "=" * 50)
    print("🎉 Setup completed!")
    
    if kaggle_ready:
        print("✅ You can now run the notebooks")
    else:
        print("⚠️  Configure Kaggle API to download the dataset")
    
    print("\n📚 Next steps:")
    print("1. Start with: BrainTumor_aproach_1.ipynb")
    print("2. Then try: Braintumor_Aproach_2.ipynb")
    print("3. Check results in BrainMRI_Results/ directories")

if __name__ == "__main__":
    main()
