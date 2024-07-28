from typing import List, Optional

from llama_cpp import Llama


# A wrapper class for the Llama language model using llama_cpp.
class LlamaCppModel:

    def load_model(self, model_path: str, n_gpu_layers: int, seed: Optional[int], n_ctx: int) -> Llama:
        """
        Load the Llmcpp model with specified parameters.

        Args:
            model_path (str): Path to the model file.
            n_gpu_layers (int): Number of GPU layers to use. -1 means use all available.
            seed (Optional[int]): Seed for random number generation.
            n_ctx (int): Size of the context window.

        Returns:
            Llama: An instance of the Llama model.
        """
        return Llama(model_path=model_path, n_gpu_layers=n_gpu_layers, seed=seed, n_ctx=n_ctx)

    # Initialize the LlamaCppModel class and load the model.
    def __init__(
        self,
        model_path: str,
        n_gpu_layers: int = -1,
        seed: Optional[int] = None,
        n_ctx: int = 2048,
    ):
        """
        model_path (str): Path to the model file.

        n_gpu_layers (int): Number of GPU layers to use. -1 means use all available. Defaults to -1. seed
        (Optional[int]): Seed for random number generation. Defaults to None. n_ctx (int): Size of the context
        window. Defaults to 2048.
        """
        self.llm = Llama(model_path=model_path, n_gpu_layers=n_gpu_layers, seed=seed, n_ctx=n_ctx)

    # Generate text based on the given prompt.
    def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        stop: Optional[List[str]] = None,
        echo: bool = False,
    ) -> dict:
        """
        Prompt (str): The input prompt for text generation. max_tokens (Optional[int]): Maximum number of
        tokens to generate. If None, generates up to the end of the context window. stop
        (Optional[List[str]]): List of strings that will stop the generation when encountered. echo (bool):
        Whether to echo the prompt in the output. Defaults to False.

        Returns:
            dict: The output from the language model.
        """

        result = self.llm(prompt, max_tokens=max_tokens, stop=stop, echo=echo)
        return result


"""
Example:
    >>> from llmrush import LlamaCppModel
    >>>
    >>> # Initialize and load the model
    >>> model = LlamaCppModel("path/to/model.gguf", n_gpu_layers=100, seed=1337, n_ctx=2048)
    >>>
    >>> # Generate text
    >>> output = model.generate(
    ...     prompt="Q: Name the planets in the solar system? A: ",
    ...     max_tokens=128,
    ...     stop=["Q:", "\n"],
    ...     echo=True
    ... )
    >>> print(output)
"""
