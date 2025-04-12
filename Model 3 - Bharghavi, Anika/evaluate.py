import torch
import torch.nn.functional as F
from nltk.translate.bleu_score import corpus_bleu

#code to add for evaluating model with saved checkpoints
def evaluate_model_with_saved_checkpoint(model_path, dataset, device=torch.device('cuda'), num_samples=5):
    print(f"Loading model from {model_path}")

    embed_size = 256
    hidden_size = 512
    vocab_size = len(dataset.vocab)
    feature_dim = embed_size

    model = ImageCaptioningModel(embed_size, hidden_size, vocab_size, feature_dim).to(device)

    #Load state dict
    model.load_state_dict(torch.load(model_path, map_location=device))
    print("Model loaded successfully!")

    #Create a small data loader for evaludation
    eval_loader = DataLoader(
        dataset,
        batch_size=num_samples,
        shuffle=True,
        num_workers=2,
        collate_fn=collate_fn,
        drop_last=False
    )

    #Calculate BLEU scores on the dataset
    bleu_scores = calculate_bleu_scores(model, eval_loader, dataset.vocab, device)
    print("BLEU Scores:")
    for metric, score in bleu_scores.items():
        print(f"{metric}: {score:.4f}")

    #Visualize some predictions
    print("\nVisualizing predictions:")
    visualize_predictions(model, eval_loader, dataset.vocab, device, num_samples)

    return model, bleu_scores