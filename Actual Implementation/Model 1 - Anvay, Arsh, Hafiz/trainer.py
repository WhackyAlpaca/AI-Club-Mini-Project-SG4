from torch import nn, optim, Tensor

class Trainer:
    def __init__(self, model, criterion, optimizer, device, padding_idx=0):
        self.device = device
        self.model = model.to(self.device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.padding_idx = padding_idx

    def generate_square_subsequent_mask(sz: int) -> Tensor:
        """Generate causal attention mask for decoder"""
        mask = torch.triu(torch.ones(sz, sz), diagonal=1).bool()
        return mask.masked_fill(mask == 1, float('-inf'))

    def create_padding_mask(self, seq: Tensor):
        """Create padding mask for sequence"""
        return (seq == self.padding_idx)

    def train_one_epoch(self, train_loader):
        self.model.train()
        total_loss = 0.0

        for images, captions in train_loader:
            images, captions = images.to(self.device), captions.to(self.device)
            
            # Generate masks
            seq_len = captions.size(1)
            tgt_mask = generate_square_subsequent_mask(seq_len).to(device)
            tgt_padding_mask = self.create_padding_mask(captions)

            # Forward pass
            self.optimizer.zero_grad()
            outputs, _ = self.model(
                images=images,
                captions=captions[:, :-1],  # Exclude last token
                tgt_mask=tgt_mask[:-1, :-1]  # Adjust mask size
            )

            # Calculate loss (compare predictions with next tokens)
            logits = outputs.transpose(1, 2)  # [batch, vocab, seq_len]
            loss = self.criterion(
                logits, 
                captions[:, 1:]  # Shift targets
            )

            # Backpropagation
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(train_loader)

    def validate(self, val_loader):
        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for images, captions in val_loader:
                images, captions = images.to(self.device), captions.to(self.device)
                
                seq_len = captions.size(1)
                tgt_mask = generate_square_subsequent_mask(seq_len).to(device)
                tgt_padding_mask = self.create_padding_mask(captions)

                outputs, _ = self.model(
                    images=images,
                    captions=captions[:, :-1],
                    tgt_mask=tgt_mask[:-1, :-1]
                )

                logits = outputs.transpose(1, 2)
                loss = self.criterion(logits, captions[:, 1:])
                total_loss += loss.item()

        return total_loss / len(val_loader)

    def run(self, train_loader, val_loader, epochs=10):
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            train_loss = self.train_one_epoch(train_loader)
            val_loss = self.validate(val_loader)
            
            print(f"Epoch {epoch+1}/{epochs} | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f}")
            
            # # Save best model
            # if val_loss < best_val_loss:
            #     best_val_loss = val_loss
            #     torch.save(self.model.state_dict(), "best_model.pth")

### TRAINING CELL ###

device = "cuda" if torch.cuda.is_available() else "cpu"

pipeline_parameters = list(single_pipeline.parameters())

optimizer = optim.Adam(pipeline_parameters, lr=0.001)  

criterion = nn.CrossEntropyLoss()  

trainer = Trainer(
    model=single_pipeline,  
    criterion=criterion,
    optimizer=optimizer,
    device=device
)

def generate_square_subsequent_mask(sz):
    mask = torch.triu(torch.ones(sz, sz), diagonal=1)
    mask = mask.masked_fill(mask == 1, float('-inf')).masked_fill(mask == 0, float(0.0))
    return mask

trainer.run(train_loader, val_loader, epochs=15)

### UNCOMMENT CODE BELOW TO SAVE MODEL PARAMETERS

# import datetime
# from IPython.display import FileLink
# current_time = datetime.datetime.now()

# torch.save(single_pipeline.state_dict(), f'model_state_{current_time}.pt')

# FileLink(f'model_state_{current_time}.pt')

# import json
# from IPython.display import FileLink
# # Save to files
# with open('/kaggle/working/word2idx.json', 'w') as f:
#     json.dump(word2idx, f)

# with open('/kaggle/working/idx2word.json', 'w') as f:
#     json.dump(idx2word, f)

# FileLink('/kaggle/working/word2idx.json')
# FileLink('/kaggle/working/idx2word.json')