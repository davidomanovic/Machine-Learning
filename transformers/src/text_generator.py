from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

def generate_text(prompt, max_length=150, temperature=0.8, top_k=50, top_p=0.95, repetition_penalty=1.2):
    """
    Generate text using GPT-2 based on the given prompt.

    Args:
        prompt (str): The input text to base the generation on.
        max_length (int): Maximum number of tokens to generate.
        temperature (float): Sampling temperature. Higher values mean more random.
        top_k (int): Top-k filtering.
        top_p (float): Top-p (nucleus) filtering.
        repetition_penalty (float): Penalty for repeating tokens.

    Returns:
        str: The generated text.
    """
    # Load pre-trained model and tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    
    # Set pad token to eos token
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.eos_token_id

    # Encode the input prompt
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    
    # Create attention mask (all ones since there's no padding)
    attention_mask = torch.ones_like(input_ids)
    
    # Generate text
    with torch.no_grad():
        output = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            do_sample=True,
            num_return_sequences=1,
            pad_token_id=tokenizer.eos_token_id,
            repetition_penalty=repetition_penalty
        )
    
    # Decode the generated tokens
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_text
