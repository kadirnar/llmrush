from typing import Optional

import torch
from llama_cpp import LlamaForCausalLM, LlamaTokenizer


class LlamaInference:
    """
    A class for performing inference with LLaMA models.

    Example:
        ```py
        # Initialize the LlamaInference object
        >>> from llmrush import LlamaInference
        >>> llama_inference = LlamaInference(model_name="path/to/llama/model")

        >>> prompt = "Question: What is the capital of France?"
        >>> response = llama_inference.generate_response(prompt)

        >>> print(response)
        ```
    """

    def __init__(self, model_name: str, device: Optional[str] = "cuda", max_new_tokens: int = 100):
        """
        Initialize the LlamaInference object.

        Args:
            model_name (str): The name or path of the LLaMA model to use.
            device (str, optional): The device to run the model on. Defaults to "cuda".
            max_new_tokens (int, optional): The maximum number of new tokens to generate. Defaults to 100.
        """
        self.model_name: str = model_name
        self.device: torch.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.max_new_tokens: int = max_new_tokens
        self.tokenizer: Optional[LlamaTokenizer] = None
        self.model: Optional[LlamaForCausalLM] = None

        # Load the model
        if self.model is None:
            self.load_model()

    def load_model(self) -> None:
        """Load the LLaMA tokenizer and model."""
        # Load the tokenizer and model
        self.tokenizer = LlamaTokenizer.from_pretrained(self.model_name)
        self.model = LlamaForCausalLM.from_pretrained(self.model_name).to(self.device)

    def generate_response(self, prompt: str, do_sample: bool = True) -> str:
        """
        Generate a response from the LLaMA model based on the given prompt.

        Args:
            prompt (str): The input prompt for the model.
            do_sample (bool, optional): Whether to use sampling for generation. Defaults to True.

        Returns:
            str: The generated response from the model.
        """
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


if __name__ == "__main__":
    # Example usage
    llama_inference = LlamaInference(model_name="meta-llama/Meta-Llama-3.1-8B-Instruct")
    prompt = "Question: What is the capital of France?"
    response = llama_inference.generate_response(prompt)
    print(response)
