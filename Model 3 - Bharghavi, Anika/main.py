#Main execution section
if __name__ == '__main__':
    #Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    #Find dataset paths
    paths = find_coco_paths()

    #Define transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    #Create datasets with limited samples
    max_train_samples = 10000 #Limit training samples
    max_val_samples = 2000 #limit validation samples

    train_dataset = COCOCaptionDataset(
        coco_dir=paths['train_images'],
        annotation_file=paths['train_annotations'],
        transform=transform,
        max_samples=max_train_samples
    )

    val_dataset = COCOCaptionDataset(
        coco_dir=paths['val_images'],
        annotation_file=paths['val_annotations'],
        transform=transform,
        max_samples=max_val_samples
    )

    #Create data loaders with smaller batch size
    batch_size = 16 
    
    def collate_fn(batch):
        first_item = batch[0]
        num_elements = len(first_item)

        if num_elements == 2:
            images, captions = zip(batch)
        elif num_elements == 3:
            images, captions, raw_captions, filenames = zip(*[(item[0], item[1], item[2]) for item in batch])
        elif num_elements == 4:
            images, captions, raw_captions, filenames = zip(*batch)
            return torch.stack(images), pad_sequence(captions, batch_first=True), raw_captions, filenames
        else:
            print(f"Warning: Each sample has {num_elements} elements, expected 2, 3 or 4")
            images = [item[0] for item in batch]
            captions =[item[1] for item in batch]

        #Stack images
        images = torch.stack(images)

        #Pad captions
        caption_lengths = [len(cap) for cap in captions]
        max_length = max(caption_lengths)

        #Initialize padded captions
        padded_captions = pad_sequence(captions, batch_first=True, padding_value=0)

        #Fill padded_captions with actual values
        for i, cap in enumerate(captions):
            end = caption_lengths[i]
            padded_captions[i, :end] = cap[:end]

        return images, padded_captions, raw_captions, filenames

    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=2, 
        collate_fn=collate_fn, 
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=2, 
        collate_fn=collate_fn, 
        drop_last=True
    )

    #Model parameters
    embed_size = 256
    hidden_size = 512
    vocab_size = len(train_dataset.vocab)
    feature_dim = embed_size #match feature dimension with embedding size

    #Initialize model
    model = ImageCaptioningModel(embed_size, hidden_size, vocab_size, feature_dim).to(device)

    #Loss and optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=train_dataset.vocab['<pad>'])
    optimizer = optim.Adam(model.parameters(), lr=3e-4)

    #Train the model
    save_dir = './model_checkpoints'
    os.makedirs(save_dir, exist_ok=True)
    print(f"Starting training with {len(train_dataset)} training samples and {len(val_dataset)} validation samples")
    training_losses, validation_losses = train_and_validate(
        model, 
        train_loader,
        val_loader, 
        criterion, 
        optimizer, 
        num_epochs=10, 
        device=device,
        save_path=save_dir
    )

    #Load best model
    best_model_path = os.path.join(save_dir, 'best_model.pth')
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path))
        print(f"Loaded best model from {best_model_path}")
    else:
        print(f"No best model found at {best_model_path}, using final model state")

    print("Visualizing some predictions...")
    visualize_predictions(model, val_loader, train_dataset.vocab, device=device, num_samples=5)

    print("Calculating BLEU scores on validation set...")
    bleu_scores = calculate_bleu_scores(model, val_loader, train_dataset.vocab, device=device)
    for metric, score in bleu_scores.items():
        print(f"{metric}: {score:.4f}")

    sample_batch = next(iter(val_loader))
    sample_image = sample_batch[0][0].to(device)

    caption, _ = predict_caption(model, sample_image, train_dataset.vocab, max_length= 20, device=device)
    print(f"Generated caption: {caption}")

    print('Training and Validation complete!')