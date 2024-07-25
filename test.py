from typing import List, Optional

from llama_cpp import Llama


class GemmaModel:
    """A wrapper class for the Gemma language model using llama_cpp."""

    def __init__(self):
        """Initialize the GemmaModel class."""
        self.llm = None

    def load_model(
            self,
            model_path: str,
            n_gpu_layers: int = -1,
            seed: Optional[int] = None,
            n_ctx: int = 2048) -> None:
        """
        Load the Gemma model with specified parameters.

        Args:
            model_path (str): Path to the model file.
            n_gpu_layers (int): Number of GPU layers to use. -1 means use all available. Defaults to -1.
            seed (Optional[int]): Seed for random number generation. Defaults to None.
            n_ctx (int): Size of the context window. Defaults to 2048.
        """
        self.llm = Llama(model_path=model_path, n_gpu_layers=n_gpu_layers, seed=seed, n_ctx=n_ctx)

    def generate(
            self,
            prompt: str,
            max_tokens: Optional[int] = None,
            stop: Optional[List[str]] = None,
            echo: bool = False) -> dict:
        """
        Generate text based on the given prompt.

        Args:
            prompt (str): The input prompt for text generation.
            max_tokens (Optional[int]): Maximum number of tokens to generate. If None, generates up to the end of the context window.
            stop (Optional[List[str]]): List of strings that will stop the generation when encountered.
            echo (bool): Whether to echo the prompt in the output. Defaults to False.

        Returns:
            dict: The output from the language model.

        Raises:
            ValueError: If the model hasn't been loaded yet.
        """
        if self.llm is None:
            raise ValueError("Model not loaded. Call load_model() first.")

        return self.llm(prompt, max_tokens=max_tokens, stop=stop, echo=echo)


# Create an instance of the GemmaModel
gemma = GemmaModel()

# Load the model
gemma.load_model(model_path="Meta-Llama-3.1-8B-Instruct-Q5_K_M.gguf", n_gpu_layers=100, seed=1337, n_ctx=2048)

# Generate text
output = gemma.generate(
    prompt="Q: Name the planets in the solar system? A: ", max_tokens=128, stop=["Q:", "\n"], echo=True)

print(output)
