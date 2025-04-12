import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from models.caption_model import ImageCaptioningModel

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10, save_path='./checkpoints'):
    training_losses = []
    validation_losses = []
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0
        for images, captions in tqdm(train_loader, desc=f"Training Epoch {epoch+1}"):
            images, captions = images.cuda(), captions.cuda()
            optimizer.zero_grad()
            outputs, regions = model(images, captions[:, :-1])

            batch_size, seq_len, vocab_size = outputs.shape
            targets = captions[:, 1:seq_len+1]
            outputs_flat = outputs.reshape(batch_size * seq_len, vocab_size)
            targets_flat = targets.reshape(batch_size * seq_len)

            word_loss = criterion(outputs_flat, targets_flat)
            region_targets = torch.zeros_like(regions.reshape(-1, regions.shape[-1])).long()
            region_loss = nn.functional.cross_entropy(regions.reshape(-1, regions.shape[-1]), region_targets.argmax(dim=1))

            loss = word_loss + 0.1 * region_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        training_losses.append(avg_train_loss)

        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for images, captions in tqdm(val_loader, desc=f"Validation Epoch {epoch+1}"):
                images, captions = images.cuda(), captions.cuda()
                outputs, regions = model(images, captions[:, :-1])

                batch_size, seq_len, vocab_size = outputs.shape
                targets = captions[:, 1:seq_len+1]
                outputs_flat = outputs.reshape(batch_size * seq_len, vocab_size)
                targets_flat = targets.reshape(batch_size * seq_len)

                word_loss = criterion(outputs_flat, targets_flat)
                region_targets = torch.zeros_like(regions.reshape(-1, regions.shape[-1])).long()
                region_loss = nn.functional.cross_entropy(regions.reshape(-1, regions.shape[-1]), region_targets.argmax(dim=1))

                loss = word_loss + 0.1 * region_loss
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        validation_losses.append(avg_val_loss)

        print(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}, Val Loss={avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), f'{save_path}/best_model.pth')
            print("Best model saved!")

    return training_losses, validation_losses
