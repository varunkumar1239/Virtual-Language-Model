import numpy as np
import pandas as pd
import random
import os
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torchvision.models as models
from torch.optim import AdamW
from transformers import BertTokenizer, BertModel

# Check for GPU
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"GPU is available: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print("GPU is not available, using CPU.")

# File paths
train_path = "/kaggle/input/vqa-dataset/archive (6)/train_data.csv"
eval_path = "/kaggle/input/vqa-dataset/archive (6)/eval_data.csv"
image_path = "/kaggle/input/vqa-dataset/archive (8)/dataset/images"

# Load and preprocess data
dataframe = pd.DataFrame(pd.read_csv(train_path))
eval_dataframe = pd.DataFrame(pd.read_csv(eval_path))
dataframe['image_id'] = dataframe['image_id'] + '.png'
eval_dataframe['image_id'] = eval_dataframe['image_id'] + '.png'

# Image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomRotation(30),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# BERT Tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
max_len = 24  # Maximum question length

# Build answer vocabulary (unchanged from original)
def build_vocab(texts, min_freq=1):
    from collections import Counter
    counter = Counter()
    for text in texts:
        tokens = tokenizer.tokenize(text)
        counter.update(tokens)
    vocab = {"<unk>": 0, "<pad>": 1, "<sos>": 2, "<eos>": 3}
    index = 4
    for word, freq in counter.items():
        if freq >= min_freq:
            vocab[word] = index
            index += 1
    return vocab

vocab_answers = build_vocab(dataframe['response'])
answers_vocab_size = len(vocab_answers)
idx2word_answers = {idx: word for word, idx in vocab_answers.items()}

# Text to tensor for answers (unchanged)
def text_to_tensor(text, vocab, max_len):
    tokens = ["<sos>"] + tokenizer.tokenize(text)[:max_len-2] + ["<eos>"]
    indices = [vocab.get(token, vocab["<unk>"]) for token in tokens]
    if len(indices) < max_len:
        indices += [vocab["<pad>"]] * (max_len - len(indices))
    else:
        indices = indices[:max_len]
    return torch.tensor(indices, dtype=torch.long)

# Custom VQA Dataset
class VQADataset(Dataset):
    def __init__(self, csv_path, image_folder, transform=None):
        self.df = pd.read_csv(csv_path)
        self.df['image_id'] = self.df['image_id'] + '.png'
        self.image_folder = image_folder
        self.transform = transform
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.max_len = 24

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = os.path.join(self.image_folder, row['image_id'])
        # Tokenize question for BERT
        question = self.tokenizer(row['question'], padding='max_length', max_length=self.max_len,
                                 truncation=True, return_tensors='pt')
        answer = text_to_tensor(row['response'], vocab_answers, 36)
        if os.path.exists(image_path):
            img = Image.open(image_path).convert("RGB")
            if self.transform:
                img = self.transform(img)
        else:
            img = torch.zeros((3, 224, 224))
            print(f"Ảnh không tồn tại: {image_path}")
        return img, question['input_ids'].squeeze(0), question['attention_mask'].squeeze(0), answer

# Create DataLoaders
train_dataset = VQADataset(train_path, image_path, transform)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
eval_dataset = VQADataset(eval_path, image_path, transform)
eval_loader = DataLoader(eval_dataset, batch_size=16, shuffle=True, num_workers=4)

# CNN Feature Extractor (unchanged)
class CNN_Feature_Extractor_pretrained(nn.Module):
    def __init__(self):
        super(CNN_Feature_Extractor_pretrained, self).__init__()
        resnet = models.resnet50(weights=None)
        weights_path = "/kaggle/input/resnet/pytorch/default/1/resnet50-11ad3fa6.pth"
        state_dict = torch.load(weights_path)
        resnet.load_state_dict(state_dict)
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
        self.fc = nn.Linear(2048, 512)

    def forward(self, x):
        x = self.feature_extractor(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        return x

# BERT-based Question Encoder
class Question_Encoder(nn.Module):
    def __init__(self):
        super(Question_Encoder, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.fc = nn.Linear(768, 512)  # Map BERT's output to 512-dim

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs[1]  # Use [CLS] token embedding
        return self.fc(pooled_output)  # Shape: [batch_size, 512]

# Attention (unchanged)
class Attention(nn.Module):
    def __init__(self, hidden_dim=512):
        super(Attention, self).__init__()
        self.attn = nn.Linear(hidden_dim * 3, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, hidden, combined_feat):
        if hidden.dim() > 2:
            hidden = hidden.squeeze(0)
        if hidden.dim() == 1:
            hidden = hidden.unsqueeze(0)
        energy = torch.tanh(self.attn(torch.cat((hidden, combined_feat), dim=1)))
        attention_weights = F.softmax(self.v(energy), dim=1)
        context = attention_weights * combined_feat
        return context, attention_weights

# Answer Decoder (unchanged)
class Answer_Decoder(nn.Module):
    def __init__(self, answer_vocab_size, embedding_size=256, hidden_dim=512, k_beam=3):
        super(Answer_Decoder, self).__init__()
        self.embedding = nn.Embedding(answer_vocab_size, embedding_size)
        self.lstm = nn.LSTM(embedding_size + 1024, hidden_dim, num_layers=3, dropout=0.2, batch_first=True)
        self.attention = Attention(hidden_dim)
        self.fc = nn.Linear(hidden_dim, answer_vocab_size)
        self.k_beam = k_beam

    def forward(self, question_feat, image_feat, answer_seq=None, answer_maxlength=36):
        combined_feat = torch.cat((question_feat, image_feat), dim=1)
        if answer_seq is not None:
            x = self.embedding(answer_seq)
            hidden_state = None
            outputs = []
            for i in range(x.size(1)):
                context, _ = self.attention(hidden_state[0][-1] if hidden_state else question_feat, combined_feat)
                lstm_input = torch.cat((x[:, i, :], context), dim=1).unsqueeze(1)
                output, hidden_state = self.lstm(lstm_input, hidden_state)
                outputs.append(self.fc(output.squeeze(1)))
            return torch.stack(outputs, dim=1)
        else:
            batch_size = combined_feat.size(0)
            device = image_feat.device
            end_token = 3
            all_results = []
            for b in range(batch_size):
                b_question_feat = question_feat[b:b+1]
                b_combined_feat = combined_feat[b:b+1]
                beams = [(torch.tensor([[2]], dtype=torch.long, device=device), 0.0, None)]
                completed_beams = []
                for _ in range(answer_maxlength):
                    candidates = []
                    for seq, score, hidden_state in beams:
                        if seq[0, -1].item() == end_token:
                            completed_beams.append((seq, score, hidden_state))
                            continue
                        x = self.embedding(seq[:, -1])
                        prev_hidden = hidden_state[0][-1] if hidden_state else b_question_feat
                        context, _ = self.attention(prev_hidden, b_combined_feat)
                        lstm_input = torch.cat((x, context), dim=1).unsqueeze(1)
                        output, new_hidden = self.lstm(lstm_input, hidden_state)
                        logits = self.fc(output.squeeze(1))
                        log_probs = F.log_softmax(logits, dim=1)
                        topk_log_probs, topk_indices = log_probs.topk(self.k_beam)
                        for i in range(self.k_beam):
                            next_token = topk_indices[:, i:i+1]
                            next_score = score + topk_log_probs[:, i].item()
                            next_seq = torch.cat([seq, next_token], dim=1)
                            candidates.append((next_seq, next_score, new_hidden))
                    if not candidates:
                        break
                    candidates.sort(key=lambda x: x[1], reverse=True)
                    beams = candidates[:self.k_beam]
                    if all(beam[0][0, -1].item() == end_token for beam in beams):
                        completed_beams.extend(beams)
                        break
                if completed_beams:
                    completed_beams.sort(key=lambda x: x[1], reverse=True)
                    best_seq = completed_beams[0][0]
                else:
                    beams.sort(key=lambda x: x[1], reverse=True)
                    best_seq = beams[0][0]
                all_results.append(best_seq)
            max_len = max(seq.size(1) for seq in all_results)
            padded_results = []
            for seq in all_results:
                if seq.size(1) < max_len:
                    padding = torch.full((1, max_len - seq.size(1)), end_token, dtype=torch.long, device=device)
                    padded_seq = torch.cat([seq, padding], dim=1)
                    padded_results.append(padded_seq)
                else:
                    padded_results.append(seq)
            return torch.cat(padded_results, dim=0)

# VQA Model with BERT
class VQA_Model(nn.Module):
    def __init__(self, answers_vocab_size, k_beam=3):
        super(VQA_Model, self).__init__()
        self.image_encoder = CNN_Feature_Extractor_pretrained().to(device)
        self.question_encoder = Question_Encoder().to(device)
        self.answer_decoder = Answer_Decoder(answers_vocab_size, k_beam=k_beam).to(device)

    def forward(self, image, input_ids, attention_mask, answer_seq=None):
        image_feat = self.image_encoder(image)
        question_feat = self.question_encoder(input_ids, attention_mask)
        output = self.answer_decoder(question_feat, image_feat, answer_seq)
        return output

# Training function (modified for BERT inputs)
def train_model(model, train_loader, eval_loader, criterion, optimizer, best_model_path, num_epochs=10, patience=5):
    import time
    model.to(device)
    best_loss = float('inf')
    no_improve_epochs = 0
    history = {"train_loss": [], "eval_loss": [], "bleu_score": []}
    os.makedirs(os.path.dirname(best_model_path), exist_ok=True)
    for epoch in range(num_epochs):
        start_time = time.time()
        model.train()
        total_loss = 0
        for images, input_ids, attention_mask, answers in train_loader:
            images, input_ids, attention_mask, answers = images.to(device), input_ids.to(device), attention_mask.to(device), answers.to(device)
            optimizer.zero_grad()
            output = model(images, input_ids, attention_mask, answers[:, :-1])
            loss = criterion(output.view(-1, output.size(-1)), answers[:, 1:].reshape(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_train_loss = total_loss / len(train_loader)
        model.eval()
        eval_loss = 0
        bleu_scores = []
        with torch.no_grad():
            for images, input_ids, attention_mask, answers in eval_loader:
                images, input_ids, attention_mask, answers = images.to(device), input_ids.to(device), attention_mask.to(device), answers.to(device)
                output = model(images, input_ids, attention_mask, answers[:, :-1])
                loss = criterion(output.view(-1, output.size(-1)), answers[:, 1:].reshape(-1))
                eval_loss += loss.item()
                predicted_answers = tensor_to_text(model(images, input_ids, attention_mask), idx2word_answers)
                answers_text = tensor_to_text(answers, idx2word_answers)
                bleu = compute_bleu(predicted_answers, answers_text)
                bleu_scores.append(bleu)
        avg_eval_loss = eval_loss / len(eval_loader)
        avg_bleu_score = sum(bleu_scores) / len(bleu_scores)
        end_time = time.time()
        print(f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {avg_train_loss:.4f}, Evaluation Loss: {avg_eval_loss:.4f}, BLEU Score: {avg_bleu_score:.4f}")
        history["train_loss"].append(avg_train_loss)
        history["eval_loss"].append(avg_eval_loss)
        history["bleu_score"].append(avg_bleu_score)
        if avg_eval_loss < best_loss:
            best_loss = avg_eval_loss
            no_improve_epochs = 0
            torch.save(model.state_dict(), best_model_path)
            print("Best model saved!")
        else:
            no_improve_epochs += 1
            print(f"No improvement for {no_improve_epochs}/{patience} epochs.")
            if no_improve_epochs >= patience:
                print("Early stopping triggered!")
                break
    return history

# BLEU score functions (unchanged)
def ngram_precision(reference, candidate, n):
    from collections import Counter
    ref_ngrams = Counter([tuple(reference[i:i+n]) for i in range(len(reference)-n+1)])
    cand_ngrams = Counter([tuple(candidate[i:i+n]) for i in range(len(candidate)-n+1)])
    overlap = sum(min(cand_ngrams[ngram], ref_ngrams.get(ngram, 0)) for ngram in cand_ngrams)
    total = sum(cand_ngrams.values())
    return overlap / total if total > 0 else 0

def brevity_penalty(reference, candidate):
    ref_len = len(reference)
    cand_len = len(candidate)
    if cand_len > ref_len:
        return 1
    else:
        return math.exp(1 - ref_len / cand_len) if cand_len > 0 else 0

def compute_bleu(reference_sentences, candidate_sentences, max_n=4):
    assert len(reference_sentences) == len(candidate_sentences)
    bleu_scores = []
    for ref, cand in zip(reference_sentences, candidate_sentences):
        precisions = [ngram_precision(ref, cand, n) for n in range(1, max_n+1)]
        geometric_mean = math.exp(sum(math.log(p) for p in precisions if p > 0) / max_n) if any(precisions) else 0
        bp = brevity_penalty(ref, cand)
        bleu_scores.append(bp * geometric_mean)
    return sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0

def tensor_to_text(tensor, idx2word):
    sentences = []
    for seq in tensor:
        words = [idx2word[idx.item()] for idx in seq if idx.item() in idx2word]
        if "<sos>" in words:
            words.remove("<sos>")
        sentence = " ".join(words).split("<eos>")[0]
        sentences.append(sentence.strip())
    return sentences

# Initialize and train model
VQA_model = VQA_Model(answers_vocab_size)
criterion = nn.CrossEntropyLoss(ignore_index=1)
optimizer = AdamW(VQA_model.parameters(), lr=1e-4, weight_decay=1e-2)
VQA_model_history = train_model(VQA_model, train_loader, eval_loader, criterion, optimizer, '/kaggle/working/VAQ_model_bert.pth', num_epochs=50)

# Test model (modified for BERT inputs)
def test_model(model, question, image_path, ground_truth, idx2word):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)
    question_inputs = tokenizer(question, padding='max_length', max_length=24, truncation=True, return_tensors='pt')
    input_ids = question_inputs['input_ids'].to(device)
    attention_mask = question_inputs['attention_mask'].to(device)
    model.eval()
    with torch.no_grad():
        output = model(image_tensor, input_ids, attention_mask)
    predicted_answer = tensor_to_text(output, idx2word)[0]
    plt.imshow(image)
    plt.axis("off")
    plt.title(f"Q: {question}\nPredicted answer: {predicted_answer}\nGround Truth: {ground_truth}", fontsize=12)
    plt.show()
    return predicted_answer

def test_random_samples(model, eval_dataframe, idx2word):
    samples = eval_dataframe.sample(n=15)
    for index, row in samples.iterrows():
        question = row['question']
        image_path = f'/kaggle/input/visual-question-answering-computer-vision-nlp/dataset/images/{row["image_id"]}.png'
        ground_truth = row['response']
        predicted_answer = test_model(model, question, image_path, ground_truth, idx2word)
test_random_samples(VQA_model, eval_dataframe, idx2word_answers)