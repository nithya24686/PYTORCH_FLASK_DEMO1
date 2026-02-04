import torch
import torch.nn as nn
import os

# Model path
MODEL_PATH = "text generator.pth"

class TextGenerator(nn.Module):
    """Simple RNN-based text generator"""
    def __init__(self, vocab_size, embedding_dim=16, hidden_dim=32):
        super(TextGenerator, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
    
    def forward(self, x):
        embedded = self.embedding(x)
        output, (hidden, cell) = self.lstm(embedded)
        logits = self.fc(output)
        return logits

def load_model(model_path):
    """Load the pre-trained text generator model"""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Load the model
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    
    # Extract model state and config if available
    if isinstance(checkpoint, dict) and 'model_state' in checkpoint:
        model_state = checkpoint['model_state']
        config = checkpoint.get('config', {})
    else:
        model_state = checkpoint
        config = {}
    
    # Initialize model with appropriate dimensions
    vocab_size = config.get('vocab_size', 5273)
    embedding_dim = config.get('embedding_dim', 16)
    hidden_dim = config.get('hidden_dim', 32)
    
    model = TextGenerator(vocab_size, embedding_dim, hidden_dim)
    model.load_state_dict(model_state)
    model.eval()
    
    return model, config

def generate_text(model, start_text, vocab, char_to_idx, idx_to_char, max_length=100, temperature=1.0):
    """Generate text using the trained model"""
    model.eval()
    
    # Convert start text to indices
    input_indices = [char_to_idx.get(char, 0) for char in start_text]
    input_tensor = torch.tensor([input_indices], dtype=torch.long)
    
    generated_text = start_text
    
    with torch.no_grad():
        for _ in range(max_length):
            output = model(input_tensor)
            # Get the last prediction
            logits = output[0, -1, :] / temperature
            probabilities = torch.softmax(logits, dim=-1)
            next_idx = torch.multinomial(probabilities, 1).item()
            
            next_char = idx_to_char.get(next_idx, '?')
            generated_text += next_char
            
            # Update input for next iteration
            input_indices.append(next_idx)
            input_tensor = torch.tensor([input_indices], dtype=torch.long)
    
    return generated_text

def main():
    """Main function to run the text generator"""
    try:
        # Load the model
        print("Loading model...")
        model, config = load_model(MODEL_PATH)
        print("Model loaded successfully!")
        print(f"Model config: {config}")
        print(f"Vocab size: 5273, Embedding dim: 16, Hidden dim: 32\n")
        
        # Create vocabulary mapping
        # Use extended ASCII and common characters
        chars = ''.join(chr(i) for i in range(32, 127)) + ''.join(chr(i) for i in range(160, 256))
        char_to_idx = {char: idx for idx, char in enumerate(chars)}
        idx_to_char = {idx: char for char, idx in char_to_idx.items()}
        
        # Pad the vocabulary to 5273
        for i in range(len(chars), 5273):
            idx_to_char[i] = '?'
        
        # Generate text with different starting prompts
        print("=" * 60)
        print("Generating text samples:")
        print("=" * 60)
        
        prompts = ["The", "Hello", "Model", "Python"]
        for prompt in prompts:
            try:
                generated = generate_text(model, prompt, None, char_to_idx, idx_to_char, max_length=50, temperature=0.8)
                print(f"\nPrompt: '{prompt}'")
                print(f"Generated: {generated}")
            except Exception as e:
                print(f"\nPrompt: '{prompt}'")
                print(f"Error generating: {e}")
        
        print("\n" + "=" * 60)
        print("Model inference completed successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
