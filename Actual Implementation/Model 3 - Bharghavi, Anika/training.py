#Training function
def train_and_validate(model, train_loader, val_loader, criterion, optimizer,
                      num_epochs=10, device=torch.device('cuda'), save_path='./'):
    #Train and validate the model
    training_losses = []
    validation_losses = []
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        #training
        model.train()
        train_loss = 0
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader))

        for i, batch in progress_bar:
                
            if len(batch) == 2:
                images, captions = batch
            elif len(batch) == 3:
                images, captions, _ = batch
            elif len(batch) == 4:
                images, captions, _, _ = batch
            else:
                print(f"Unexpected batch structure with {len(batch)} elements")
                images = batch[0]
                captions = batch[1]

            images, captions = images.to(device), captions.to(device)

            #zero the gradients
            optimizer.zero_grad()

            #forward pass
            outputs = model(images, captions[:, :-1]) #input without END token

            batch_size = outputs.size(0)
            seq_len = outputs.size(1)
            vocab_size = outputs.size(2)
            
            outputs_flat = outputs.reshape(-1, vocab_size)
            targets_flat = captions[:, 1:seq_len+1].reshape(-1)

            #Double check sizes before loss calculation
            if outputs_flat.size(0) != targets_flat.size(0):
                print(f"Critical warning: Sizes don't match after reshape.")
                print(f"Original shapes - Outputs: {outputs.shape}, Targets: {targets.shape}")
                print(f"Flat shapes - Outputs: {outputs_flat.shape}, Targets: {targets_flat.shape}")
                
                #Resize to smaller size if needed
                min_size = min(outputs_flat.size(0), targets_flat.size(0))
                outputs_flat = outputs_flat[:min_size]
                targets_flat = targets_flat[:min_size]

            #calculate loss
            loss = criterion(outputs_flat, targets_flat)

            #Backward pass and optimize
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5) #clip gradients
            optimizer.step()

            #Update running loss
            train_loss += loss.item()

            #Upate progress bar
            progress_bar.set_description(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")

        avg_train_loss = train_loss / len(train_loader)
        training_losses.append(avg_train_loss)
        print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {avg_train_loss:.4f}")

    #Validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        val_progress_bar = tqdm(enumerate(val_loader), total=len(val_loader))
        for i, batch in val_progress_bar:
                
            if len(batch) == 2:
                images, captions = batch
            elif len(batch) == 3:
                images, captions, _ = batch
            elif len(batch) == 4:
                images, captions, _, _ = batch
            else:
                print(f"Unexpected validation batch structure with {len(batch)} elements")
                images = batch[0]
                captions = batch[1]

            images, captions = images.to(device), captions.to(device)

            #Forward pass
            outputs = model(images, captions[:, :-1])

            #Reshape for loss calculation
            batch_size = outputs.size(0)
            seq_len = outputs.size(1)
            vocab_size = outputs.size(2)

            outputs_flat = outputs.reshape(-1, vocab_size)
            targets_flat = captions[:, 1:seq_len+1].reshape(-1)

            #Double check sizes before loss calculation
            if outputs_flat.size(0) != targets_flat.size(0):
                min_size = min(outputs_flat.size(0), targets_flat.size(0))
                outputs_flat = outputs_flat[:min_size]
                targets_flat = targets_flat[:min_size]
                
            #Calculate loss
            loss = criterion(outputs_flat, targets_flat)
            val_loss += loss.item()

            #Update validation loss
            val_progress_bar.set_description(f"Validation, Loss: {loss.item():.4f}")

        avg_val_loss = val_loss / len(val_loader)
        validation_losses.append(avg_val_loss)
        print(f"Epoch {epoch+1}/{num_epochs}, Validation Loss: {avg_val_loss:.4f}")
    
        #Save model if validation loss improves
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            #Create directory if it doesnt exist
            os.makedirs(save_path, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(save_path, 'best_model.pth'))
            print(f"Saved new best model with validation loss: {avg_val_loss:.4f}")

    
    os.makedirs(save_path, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(save_path, 'final_model.pth'))
    print(f"Saved final model after {num_epochs} epochs")

    return training_losses, validation_losses