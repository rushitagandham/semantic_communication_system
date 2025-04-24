## 🔁 Full Setup (Bash Script)

If you're on Linux or macOS (or using Git Bash on Windows), you can run the entire project pipeline in one go using this script:

```bash
# Clone the repository
git clone https://github.com/your-username/semantic_communication_system.git
cd semantic_communication_system

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate        # On macOS/Linux
# .venv\Scripts\activate         # On Windows PowerShell

# Install required Python packages
pip install --upgrade pip
pip install -r requirements.txt

# Download AIS data (zipped and extracted)
python scripts/download_ais_data.py

# Generate structured maritime sentences from AIS data
python scripts/generate_sentences.py

# Build the tokenizer vocabulary from generated sentences
python utils/build_vocab.py

# Train the semantic communication model
python scripts/train.py

# Evaluate model performance using BLEU and Semantic Similarity
python scripts/evaluate.py
