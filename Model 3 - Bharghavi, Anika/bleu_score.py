#Function to calculate BLEU scores
def calculate_bleu_scores(model, data_loader, vocab, device='cuda'):
    
    model.eval()
    references = []
    hypotheses = []

    print("Calculating BLEU scores...")
    itos = {idx: word for word, idx in vocab.items()}
    with torch.no_grad():
        for batch in tqdm(data_loader):
            images, captions_tensor, captions_raw, _ = batch
            #Handle different batch structures
            if len(batch) == 2:
                images, captions_raw = batch
            elif len(batch) == 3:
                images, captions_raw, _ = batch
            elif len(batch) == 4:
                images, _, captions_raw, _ = batch
            else:
                print(f"Unexpected batch structure with {len(batch)} elements")
                images = batch[0]
                captions_raw = batch[2] if len(batch) > 2 else batch[1]

            #Process each sample in the batch
            for i in range(images.size(0)):
                image = images[i].to(device)

                #Get reference caption (convert from raw string or tensor)
                if isinstance(captions_raw[i], torch.Tensor):
                    reference = []
                    for idx in captions_tensor[i].tolist():
                        if idx == vocab['<pad>'] or idx == vocab['<start>']:
                            continue
                        if idx == vocab['<end>']:
                            break
                        reference.append(itos[idx])

                else:
                    #If it's already a string
                    reference = nltk.word_tokenize(captions_tensor[i].lower())

                #Generate caption
                predicted_caption, _ = predict_caption(model, image, vocab, max_length=20, device=device)
                hypothesis = nltk.word_tokenize(predicted_caption.lower())

                #Add to lists for BLEU calculation
                references.append([reference])
                hypotheses.append(hypothesis)

    #Calculate BLEU scores
    bleu1 = corpus_bleu(references, hypotheses, weights=(1, 0, 0, 0))
    bleu2 = corpus_bleu(references, hypotheses, weights=(0.5, 0.5, 0, 0))
    bleu3 = corpus_bleu(references, hypotheses, weights=(0.33, 0.33, 0.33, 0))
    bleu4 = corpus_bleu(references, hypotheses, weights=(0.25, 0.25, 0.25, 0.25))

    return {
        'BLEU-1': bleu1,
        'BLEU-2': bleu2,
        'BLEU-3': bleu3,
        'BLEU-4': bleu4
    }