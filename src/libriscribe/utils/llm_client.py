# src/libriscribe/utils/llm_client.py
import openai
from openai import OpenAI  # For OpenAI
import logging
from tenacity import retry, stop_after_attempt, wait_random_exponential
from libriscribe.settings import Settings

import anthropic  # For Claude
from google import genai # For Google AI Studio (NEW SDK)
# Import types for configuration if needed later, though dict works fine for now
# from google.genai import types
import requests  # For DeepSeek and Mistral

# ADDED THIS: Import the function
from libriscribe.utils.file_utils import extract_json_from_markdown

logger = logging.getLogger(__name__)

# Configure httpx logger to be less verbose
httpx_logger = logging.getLogger("httpx")
httpx_logger.setLevel(logging.WARNING)  # Or ERROR, to suppress even warnings

class LLMClient:
    """Unified LLM client for multiple providers."""

    def __init__(self, llm_provider: str):
        self.settings = Settings()
        self.llm_provider = llm_provider
        self.client = self._get_client()  # Initialize the correct client
        self.model = self._get_default_model()

    def _get_client(self):
        """Initializes the appropriate client based on the provider."""
        if self.llm_provider == "openai":
            if not self.settings.openai_api_key:
                raise ValueError("OpenAI API key is not set.")
            return OpenAI(api_key=self.settings.openai_api_key)
        elif self.llm_provider == "claude":
            if not self.settings.claude_api_key:
                raise ValueError("Claude API key is not set.")
            return anthropic.Anthropic(api_key=self.settings.claude_api_key)
        elif self.llm_provider == "google_ai_studio":
            if not self.settings.google_ai_studio_api_key:
                raise ValueError("Google AI Studio API key is not set in settings.")
            # --- FIX: Pass the API key explicitly ---
            try:
                return genai.Client(api_key=self.settings.google_ai_studio_api_key)
            except Exception as e:
                 # Log the specific error during client initialization
                logger.error(f"Failed to initialize Google GenAI Client: {e}")
                # Re-raise the exception or handle it as appropriate
                raise ValueError(f"Google GenAI Client initialization failed. Check API key and permissions. Error: {e}")

        elif self.llm_provider == "deepseek":
             if not self.settings.deepseek_api_key:
                raise ValueError("DeepSeek API key is not set.")
             return None  # No client object, we'll use requests directly
        elif self.llm_provider == "mistral":
             if not self.settings.mistral_api_key:
                raise ValueError("Mistral API key is not set")
             return None
        else:
            raise ValueError(f"Unsupported LLM provider: {self.llm_provider}")

    # ... (rest of the LLMClient class, including _get_default_model and generate_content) ...

    def _get_default_model(self):
        """Gets the default model name for the selected provider."""
        if self.llm_provider == "openai":
            return "gpt-4o-mini"
        elif self.llm_provider == "claude":
            return "claude-3-opus-20240229" # Or another appropriate Claude 3 model
        elif self.llm_provider == "google_ai_studio":
            return "gemini-2.0-flash" # Updated model name for new SDK
        elif self.llm_provider == "deepseek":
             return "deepseek-coder-6.7b-instruct"
        elif self.llm_provider == "mistral":
            return "mistral-medium-latest"
        else:
            return "unknown"  # Should not happen, but good for safety

    def set_model(self, model_name: str):
      self.model = model_name

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
    def generate_content(self, prompt: str, max_tokens: int = 2000, temperature: float = 0.7, language: str = "English") -> str:
        """
        Generates text using the selected LLM provider.
        Now supports specifying the output language explicitly.
        """
        try:
            # Append language instruction to prompt if not already included
            if "IMPORTANT: The content should be written entirely in" not in prompt and language != "English":
                prompt += f"\n\nIMPORTANT: Generate the response in {language}."

            if self.llm_provider == "openai":
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                return response.choices[0].message.content.strip()

            elif self.llm_provider == "claude":
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    messages=[{"role": "user", "content": prompt}]
                )
                # Check if content is a list and not empty before accessing index 0
                if response.content and isinstance(response.content, list) and len(response.content) > 0:
                     # Ensure the first element has a 'text' attribute
                     if hasattr(response.content[0], 'text'):
                         return response.content[0].text.strip()
                     else:
                         logger.error("Claude response content item missing 'text' attribute.")
                         return "" # Or handle appropriately
                else:
                     logger.error("Unexpected Claude response format or empty content.")
                     return "" # Or handle appropriately


            elif self.llm_provider == "google_ai_studio":
                # Use the new SDK's client method
                generation_config = {
                    "max_output_tokens": max_tokens,
                    "temperature": temperature,
                }
                # Use the client object initialized in _get_client
                response = self.client.models.generate_content(
                    model=f'models/{self.model}', # Model name needs 'models/' prefix for new SDK client
                    contents=prompt,
                    generation_config=generation_config # Pass config dict here
                )
                # Add error handling for potentially blocked responses or missing text
                if response and hasattr(response, 'text'):
                    return response.text.strip()
                elif response and response.prompt_feedback and response.prompt_feedback.block_reason:
                    logger.warning(f"Google GenAI response blocked. Reason: {response.prompt_feedback.block_reason}")
                    # Return a message indicating blockage, or handle differently
                    return f"[Blocked by Safety Filter: {response.prompt_feedback.block_reason}]"
                elif response and not response.candidates: # Handle cases where response is generated but candidates list is empty
                    logger.warning(f"Google GenAI response finished with reason: {response.prompt_feedback.block_reason if response.prompt_feedback else 'Unknown'}")
                    return f"[Content Generation Stopped: {response.prompt_feedback.block_reason if response.prompt_feedback else 'Unknown Reason'}]"

                else:
                    # Log the full response if text is missing unexpectedly for debugging
                    logger.error(f"Google GenAI response missing text or blocked without clear reason. Full response: {response}")
                    return "" # Return empty string on error or blocked content


            elif self.llm_provider == "deepseek":
                headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.settings.deepseek_api_key}"
                }
                data = {
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": max_tokens,
                    "temperature": temperature
                }
                response = requests.post("https://api.deepseek.com/v1/chat/completions", headers=headers, json=data, timeout=120) # Timeout
                response.raise_for_status() # Raise for HTTP errors
                return response.json()["choices"][0]["message"]["content"].strip()

            elif self.llm_provider == "mistral":
                headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.settings.mistral_api_key}"
                }
                data = {
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": max_tokens,
                    "temperature": temperature
                }

                response = requests.post("https://api.mistral.ai/v1/chat/completions", headers=headers, json=data, timeout=120)
                response.raise_for_status()
                return response.json()['choices'][0]['message']['content'].strip()

            else:
                logger.error(f"Attempted to generate content with unsupported provider: {self.llm_provider}")
                return ""

        except Exception as e:
            logger.exception(f"Error during {self.llm_provider} API call: {e}")
            print(f"ERROR: {self.llm_provider} API error: {e}")
            return "" # Return empty string on failure after retries

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(3))
    def generate_content_with_json_repair(self, original_prompt: str, max_tokens:int = 2000, temperature:float=0.7) -> str:
        """Generates content and attempts to repair JSON errors."""
        response_text = self.generate_content(original_prompt, max_tokens, temperature)
        if response_text:
            # First, try to extract JSON directly
            json_data = extract_json_from_markdown(response_text)
            if json_data is not None:
                # If successful, return the original response (might contain markdown)
                return response_text

            # If direct extraction failed, attempt repair
            else:
                # Check if the response *looks* like it should contain JSON but is broken
                if '{' in response_text and '}' in response_text:
                    logger.warning("Initial response did not contain valid JSON within markdown. Attempting repair.")
                    repair_prompt = f"""
                    You are an AI assistant that ONLY outputs valid JSON.
                    The following text contains malformed JSON. Fix it and return ONLY the corrected JSON object.
                    Do NOT include any explanations or markdown formatting.

                    Broken Text:
                    ---
                    {response_text}
                    ---

                    Corrected JSON:
                    """
                    # Use low temperature for deterministic correction
                    repaired_response = self.generate_content(repair_prompt, max_tokens=max_tokens, temperature=0.1)

                    if repaired_response:
                        # Try extracting JSON from the *repaired* response
                        repaired_json_data = extract_json_from_markdown(repaired_response)
                        if repaired_json_data is not None:
                            logger.info("JSON repair successful.")
                            return repaired_response
                        else:
                            logger.error("JSON repair attempt failed to produce valid JSON.")
                            return response_text # Return original broken text if repair fails
                    else:
                         logger.error("JSON repair prompt failed to generate a response.")
                         return response_text # Return original broken text
                else:
                     # If the original response didn't look like JSON, return it as is.
                    # Check if it was blocked by safety filter
                    if "[Blocked by Safety Filter" in response_text or "[Content Generation Stopped" in response_text:
                         logger.warning(f"Content generation was blocked or stopped: {response_text}")
                    else:
                         logger.info("Initial response did not appear to contain JSON. Returning as is.")
                    return response_text
        else:
            # If the initial generation failed
            logger.error("Initial content generation failed for JSON repair.")
            return "" # Return empty string if initial generation fails
