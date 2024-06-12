import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM

def calculate_tokenizer_length(file_path, model_name='llama'):

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()

    encoded_text = tokenizer.encode(text)

    tokenizer_length = len(encoded_text)

    return tokenizer_length

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Calculate tokenizer length of a text file.")
    
    parser.add_argument('file_path', type=str, help='Path to the TXT file.')
    
    parser.add_argument('--model_name', type=str, default='gpt2', help='Pretrained model name (default: gpt2).')
    
    args = parser.parse_args()
    
    tokenizer_length = calculate_tokenizer_length(args.file_path, args.model_name)
    print(f"The tokenizer length of the text in '{args.file_path}' is: {tokenizer_length}")
