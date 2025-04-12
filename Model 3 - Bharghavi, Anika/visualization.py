#Visualization function
def visualize_predictions(model, val_loader, vocab, device=torch.device('cuda'), num_samples=5):
    
    #visualize some example predictions
    model.eval()

    #get samples for visualization
    samples = []
    
    with torch.no_grad():
        for batch in val_loader:
            if len(batch) == 4:
                images, captions_tensor, captions_raw, filenames = batch
            
            for i in range(min(num_samples - len(samples), len(images))):
                #get image
                image=images[i]

                #get reference caption
                if isinstance(captions_raw[i], torch.Tensor):
                    ref_caption = []
                    for idx in captions_raw[i].tolist():
                        if idx == vocab['<pad>'] or idx == vocab['<start>']:
                            continue
                        if idx == vocab['<end>']:
                            break
                        ref_caption.append(itos[idx])
                    ref_caption = ' '.join(ref_caption)

                else:
                    #If it's already a string
                    ref_caption = captions_raw[i]

                #generate caption
                pred_caption, attention_weights = predict_caption(model, image, vocab, device=device)

                samples.append((image, ref_caption, pred_caption, attention_weights, filenames[i]))

                if len(samples) >= num_samples:
                    break
            if len(samples) >= num_samples:
                break

    #Calculate BLEU score for each sample
    sample_bleu_scores = []
    for _, ref, pred, _, _ in samples:
        reference = [nltk.word_tokenize(ref.lower())]
        hypothesis = nltk.word_tokenize(pred.lower())
        bleu1 = sentence_bleu(reference, hypothesis, weights=(1, 0, 0, 0))
        bleu4 = sentence_bleu(reference, hypothesis, weights=(0.25, 0.25, 0.25, 0.25))
        sample_bleu_scores.append((bleu1, bleu4))
        