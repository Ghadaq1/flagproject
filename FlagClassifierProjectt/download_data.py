import os
import zipfile

# Optional: Set environment variable for custom location of kaggle.json
os.environ['KAGGLE_CONFIG_DIR'] = os.path.join(os.getcwd(), '.kaggle')

# Download dataset
print("Downloading dataset...")
os.system("kaggle datasets download -d sjetley/country-flags-in-the-wild")

# Unzip the dataset
with zipfile.ZipFile("country-flags-in-the-wild.zip", 'r') as zip_ref:
    zip_ref.extractall("dataset")

print("Dataset downloaded and extracted to 'dataset/' folder.")
