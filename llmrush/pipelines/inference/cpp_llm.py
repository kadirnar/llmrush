import torch
from llama_cpp import LlamaTokenizer, LlamaForCausalLM
from colorama import Fore, Style, init

# Initialize
init()


class LlamaInference:
    def __init__(
        self, model_name: str, device: str = "cuda", max_new_tokens: int = 100
    ):
        self.model_name = model_name
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.max_new_tokens = max_new_tokens

        self.load_model()

    def load_model(self):
        # Load the tokenizer and model
        self.tokenizer = LlamaTokenizer.from_pretrained(self.model_name)
        self.model = LlamaForCausalLM.from_pretrained(self.model_name).to(self.device)

    def generate_response(self, prompt: str, do_sample: bool = True) -> str:
        # Tokenize the input prompt
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)

        # Generate output from the model
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                max_length=input_ids.shape[-1] + self.max_new_tokens,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        # Decode the output tokens to text
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response


# Usage example
if __name__ == "__main__":
    model_name = "llama-7b"
    llm = LlamaInference(model_name=model_name, device="cuda")

    while True:
        prompt = input(
            f"\n{Fore.BLUE}Enter prompt (or 'quit' to exit): {Style.RESET_ALL}"
        )
        if prompt.lower() == "quit":
            break

        response = llm.generate_response(prompt)
        print(f"{Fore.YELLOW}Response:{Style.RESET_ALL} {response}")
