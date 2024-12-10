from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

def generate_text(prompt, max_length=100, temperature=0.7, top_k=50, top_p=0.95):
    """
    Generate text using GPT-2 based on the given prompt.
    Input:
        prompt (str): The input text to base the generation on.
        max_length (int): Maximum number of tokens to generate.
        temperature (float): Sampling temperature. Higher values mean more random.
        top_k (int): Top-k filtering.
        top_p (float): Top-p (nucleus) filtering.
    Return:
        str: The generated text.
    """
    # Load pre-trained model and tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2LMHeadModel.from_pretrained('gpt2')

    
    input_ids = tokenizer.encode(prompt, return_tensors='pt') # Encode the input prompt

    # Generate text
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            do_sample=True,
            num_return_sequences=1
        )

    # Decode the generated tokens
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_text

if __name__ == "__main__":
    user_prompt = input("Enter a prompt: ")
    generated = generate_text(user_prompt)
    print("\nGenerated Text:\n")
    print(generated)
