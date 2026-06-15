

"""

setup_dependencies.py
--------------------

A reusable utility module for installing dependencies and generating a requirements.txt file
for reproducibility ML experiments


Usage:
    python src/setup_dependencies.py
"""



import subprocess
import sys
import pathlib import Path



#Depencies list
DEPENDENCIES = [
    "pandas",
    "numpy",
    "scikit-learn",
    "matplotlib",
    "seaborn",
    "xgboost",
    "jupyter",
    "streamlit"
]


REQUIREMENTS_CONTENT = """\
pandas>=1.5.0
numpy>=1.21.0
scikit-learn>=1.0.0
matplotlib>=3.5.0
seaborn>=0.11.0
xgboost>=1.5.0
streamlit>=1.12.0
"""



def install_dependencies():
  """
  Install all listed dependencies using pip.
  """
  print("## 📦 Installing Dependencies ##")
  for package in DEPENDENCIES:
       print(f"Installing: {package} ...")
       subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", package])
  print("✅ All dependencies installed successfully.\n")


def create_requirements_txt(output_path: str = "requirements.txt"):
    """

    Creates (or overwrites) a requirements.txt file based with pinned versions.
    """
    req_path = Path(output_path)
    req_path.write_text(REQUIREMENTS_CONTENT.strip())
    print(f"📝 requirements.txt created at: {req_path.resolve()}\n")


def set_environment():
  """

  Installs dependencies and generates a requirements.txt file with pinned versions.
  """

  install_dependencies()
  create_requirements_txt()


#Allow direct execution
if __name__ == "__main__":
    setup_environment()
