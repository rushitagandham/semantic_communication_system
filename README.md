
#!/bin/bash

# -------------------------------
# Semantic Maritime Communication System
# -------------------------------
echo "🔧 Setting up project..."

# Clone repo
git clone https://github.com/your-username/semantic_communication_system.git
cd semantic_communication_system

# Set up virtual environment
echo "🐍 Creating virtual environment..."
python -m venv .venv
source .venv/bin/activate || source .venv/Scripts/activate

# Install dependencies
echo "📦 Installing packages..."
pip install --upgrade pip
pip install -r requirements.txt

# -------------------------------
# Data Processing
# -------------------------------

# Download raw AIS data
echo "🌍 Downloading AIS data..."
python scripts/download_ais_data.py

# Generate maritime sentence data
echo "✍️  Generating training sentences..."
python scripts/generate_sentences.py

# Build tokenizer vocabulary
echo "🔤 Building tokenizer vocabulary..."
python utils/build_vocab.py

# -------------------------------
# Model Training
# -------------------------------
echo "🧠 Starting model training..."
python scripts/train.py

# -------------------------------
# Evaluation
# -------------------------------
echo "🧪 Running evaluation..."
python scripts/evaluate.py

echo "✅ Done!"
