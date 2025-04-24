import sys
import os

# Add parent directory (project root) to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from utils.tokenizer import SimpleTokenizer
from models.full_model import SemanticCommSystem
from tqdm import tqdm

# ------------ CONFIG ------------ #
DATA_PATH = "data/processed/maritime_sentences.txt"
VOCAB_PATH = "data/processed"
BATCH_SIZE = 32
EPOCHS = 15
MAX_LEN = 30
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# -------------------------------- #

# Custom Dataset
class MaritimeDataset(Dataset):
    def __init__(self, sentences, tokenizer):
        self.sentences = sentences
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        encoded = self.tokenizer.encode(self.sentences[idx])[:MAX_LEN]
        return torch.tensor(encoded), torch.tensor(encoded)

def collate_fn(batch):
    src, tgt = zip(*batch)
    src = pad_sequence(src, batch_first=True, padding_value=0)
    tgt = pad_sequence(tgt, batch_first=True, padding_value=0)
    return src.to(DEVICE), tgt.to(DEVICE)

# Load data + tokenizer
with open(DATA_PATH, "r", encoding="utf-8") as f:
    lines = f.read().splitlines()
    lines = lines[:200000]  # just 1k examples


tokenizer = SimpleTokenizer()
tokenizer.load(VOCAB_PATH)

dataset = MaritimeDataset(lines, tokenizer)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

# Init model
model = SemanticCommSystem(vocab_size=len(tokenizer.word2idx)).to(DEVICE)
optimizer = Adam(model.parameters(), lr=1e-4)
criterion = CrossEntropyLoss(ignore_index=0)

# -------- Training Loop -------- #
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for src_tokens, tgt_tokens in tqdm(dataloader, desc=f"Epoch {epoch+1}"):
        logits = model(src_tokens, tgt_tokens[:, :-1])
        loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_tokens[:, 1:].reshape(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1} Loss: {total_loss / len(dataloader):.4f}")

torch.save(model.state_dict(), "model_checkpoint.pt")
print("✅ Model saved as model_checkpoint.pt")