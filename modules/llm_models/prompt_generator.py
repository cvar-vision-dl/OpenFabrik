import re
import os
import subprocess
import time
from typing import List, Tuple, Optional, Dict
from pathlib import Path
from ollama import chat, ChatResponse, Client


class PromptGenerator:
    """
    A class for generating image prompts and extracting classes using Ollama models.

    Supports two strategies:
    1. Batch: Traditional method asking for all prompts at once
    2. Iterative: Ask for prompts one by one while maintaining conversation context
    """

    def __init__(self, model_name: str = 'cogito:latest'):
        """
        Initialize the PromptGenerator.

        Args:
            model_name: Name of the Ollama model to use
        """
        self.model_name = model_name
        self.system_prompt = ""
        self.project_info = ""
        self.last_response = ""
        self.prompts = []
        self.classes = []
        self.conversation_history = []  # For iterative mode
        self.ollama_client = Client(timeout=240)

    def read_file(self, filepath: str) -> str:
        """
        Read content from a text file.

        Args:
            filepath: Path to the file to read

        Returns:
            Content of the file as string

        Raises:
            FileNotFoundError: If the file doesn't exist
            IOError: If there's an error reading the file
        """
        try:
            path = Path(filepath)
            if not path.exists():
                raise FileNotFoundError(f"File not found: {filepath}")

            with open(path, 'r', encoding='utf-8') as file:
                content = file.read().strip()

            print(f"Successfully read file: {filepath} ({len(content)} characters)")
            return content

        except Exception as e:
            raise IOError(f"Error reading file {filepath}: {str(e)}")

    def load_prompts(self, system_prompt_file: str, project_info_file: str) -> None:
        """
        Load system prompt and project info from files.

        Args:
            system_prompt_file: Path to system prompt file
            project_info_file: Path to project info file
        """
        print("Loading prompt files...")
        self.system_prompt = self.read_file(system_prompt_file)
        self.project_info = self.read_file(project_info_file)
        print("Prompt files loaded successfully!")

    def combine_prompts(self) -> str:
        """
        Combine system prompt and project info into a single prompt.

        Returns:
            Combined prompt string

        Raises:
            ValueError: If prompts haven't been loaded
        """
        if not self.system_prompt or not self.project_info:
            raise ValueError("System prompt and project info must be loaded first. Call load_prompts().")

        # Combine the prompts following the original format
        combined_prompt = (
            f"** SYSTEM PROMPT **\n"
            f"{self.system_prompt}\n\n"
            f"** PROJECT INFO **\n"
            f"{self.project_info}"
        )

        return combined_prompt

    def _start_ollama_server(self):
        """Start the Ollama server."""
        try:
            print(f'Launching Ollama server...')
            os.system(f"pkill ollama")
            process = subprocess.Popen(["ollama", "serve"],
                                       stdout=subprocess.DEVNULL,
                                       stderr=subprocess.DEVNULL
                                       )
            time.sleep(5)
            os.system(f"ollama stop {self.model_name}")
            print(f'Successfully launched!')
        except Exception as e:
            print(f"Warning: Could not manage Ollama server: {e}")

    def unload_model(self):
        os.system(f"ollama stop {self.model_name}")
        print(f"INFO: Ollama server stopped")

    def send_to_model(self, prompt: str) -> ChatResponse:
        """
        Send prompt to the Ollama model and get response.

        Args:
            prompt: The prompt to send to the model

        Returns:
            ChatResponse object from Ollama

        Raises:
            Exception: If there's an error communicating with the model
        """
        try:
            self._start_ollama_server()

            print(f"Sending prompt to model: {self.model_name}")
            print(f"Prompt length: {len(prompt)} characters")

            try:
                response = self.ollama_client.chat(
                    model=self.model_name,
                    messages=[{
                        'role': 'user',
                        'content': prompt,
                    }],
                    think=False
                )
            except:
                response = {
                    "message": {
                        "content": ""
                    }
                }

            print("Response received from model!")
            return response

        except Exception as e:
            raise Exception(f"Error communicating with model {self.model_name}: {str(e)}")

    def send_conversation_to_model(self, messages: List[Dict]) -> ChatResponse:
        """
        Send a conversation (multiple messages) to the Ollama model.

        Args:
            messages: List of message dictionaries with 'role' and 'content'

        Returns:
            ChatResponse object from Ollama
        """
        try:
            print(f"Sending conversation to model: {self.model_name}")
            print(f"Conversation length: {len(messages)} messages")

            try:
                response = self.ollama_client.chat(
                    model=self.model_name,
                    messages=messages,
                    think=False
                )
            except:
                response = {
                    "message": {
                        "content": ""
                    }
                }

            print("Response received from model!")
            return response

        except Exception as e:
            raise Exception(f"Error communicating with model {self.model_name}: {str(e)}")

    def parse_prompts(self, text: str) -> List[str]:
        """
        Extract numbered prompts from text enclosed in <prompts> tags.

        Args:
            text: Raw text response from the model

        Returns:
            List of extracted prompts
        """
        # First extract content inside <prompts>...</prompts>
        match = re.search(r'<prompts>(.*?)</prompts>', text, re.DOTALL | re.IGNORECASE)
        if not match:
            print("No <prompts> section found in response")
            return []

        inner_text = match.group(1).strip()

        # Pattern to match numbered prompts like "1. ..." or "1) ..."
        prompt_pattern = r'^\s*(\d+)[\.\)]\s+(.+?)(?=^\s*\d+[\.\)]|\Z)'
        matches = re.findall(prompt_pattern, inner_text, re.MULTILINE | re.DOTALL)

        prompts = []
        for _, content in matches:
            cleaned = re.sub(r'\s+', ' ', content.strip())
            prompts.append(cleaned)

        print(f"Extracted {len(prompts)} prompts from <prompts> section")
        return prompts

    def parse_single_prompt(self, text: str) -> Optional[str]:
        """
        Extract a single prompt from text. Looks for <prompt> tags or plain text.

        Args:
            text: Raw text response from the model

        Returns:
            Extracted prompt or None
        """
        # First try to find content in <prompt>...</prompt> tags
        match = re.search(r'<prompt>(.*?)</prompt>', text, re.DOTALL | re.IGNORECASE)
        if match:
            prompt = re.sub(r'\s+', ' ', match.group(1).strip())
            print(f"Extracted prompt from <prompt> tags: {prompt[:50]}...")
            return prompt

        # If no tags, try to extract the main content (skip common responses)
        text = text.strip()

        # Skip common non-prompt responses
        skip_patterns = [
            r'^(sure|of course|here\'s|another|okay)',
            r'^(i can|i\'ll|let me)',
            r'^(here is|here\'s another)',
        ]

        for pattern in skip_patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE).strip()

        # Remove quotes if the entire text is quoted
        if text.startswith('"') and text.endswith('"'):
            text = text[1:-1].strip()

        if text and len(text) > 10:  # Reasonable prompt length
            prompt = re.sub(r'\s+', ' ', text)
            print(f"Extracted prompt from plain text: {prompt[:50]}...")
            return prompt

        print("No valid prompt found in response")
        return None

    def parse_classes(self, text: str) -> List[str]:
        """
        Extract classes from the model response using <classes> tags.

        Args:
            text: Raw text response from the model

        Returns:
            List of extracted classes
        """
        # Pattern to match content within <classes></classes> tags
        classes_pattern = r'<classes>(.*?)</classes>'

        match = re.search(classes_pattern, text, re.DOTALL | re.IGNORECASE)

        if match:
            classes_content = match.group(1).strip()
            # Split by periods and clean up each class
            classes = [cls.strip() for cls in classes_content.split('.') if cls.strip()]
            print(f"Extracted {len(classes)} classes from response")
            return classes
        else:
            print("No classes found in response (no <classes> tags detected)")
            return []

    def parse_response(self, response: ChatResponse) -> Tuple[List[str], List[str]]:
        """
        Parse the model response to extract prompts and classes.

        Args:
            response: ChatResponse object from Ollama

        Returns:
            Tuple of (prompts_list, classes_list)
        """
        # Extract the text content from the response
        response_text = response['message']['content']
        self.last_response = response_text

        print("Parsing response...")
        print(f"Response length: {len(response_text)} characters")

        # Extract prompts and classes
        prompts = self.parse_prompts(response_text)
        classes = self.parse_classes(response_text)

        # Store results
        self.prompts = prompts
        self.classes = classes

        return prompts, classes

    def generate_prompts_and_classes(self,
                                     system_prompt_file: str,
                                     project_info_file: str) -> Tuple[List[str], List[str]]:
        """
        Complete workflow: load files, generate prompts, and extract data.
        This is the traditional batch method.

        Args:
            system_prompt_file: Path to system prompt file
            project_info_file: Path to project info file

        Returns:
            Tuple of (prompts_list, classes_list)
        """
        # Load the prompt files
        self.load_prompts(system_prompt_file, project_info_file)

        # Combine prompts
        combined_prompt = self.combine_prompts()

        # Send to model
        response = self.send_to_model(combined_prompt)

        # Parse response
        prompts, classes = self.parse_response(response)

        return prompts, classes

    def generate_prompts_iteratively(self,
                                     system_prompt_file: str,
                                     project_info_file: str,
                                     num_prompts: int,
                                     execution_id: int = 1,
                                     classes: bool = True) -> Tuple[List[str], List[str]]:
        """
        Generate prompts iteratively, asking for one prompt at a time.

        Args:
            system_prompt_file: Path to system prompt file
            project_info_file: Path to project info file
            num_prompts: Number of prompts to generate
            execution_id: Execution identifier for logging

        Returns:
            Tuple of (prompts_list, classes_list)
        """
        print(f"\n=== ITERATIVE PROMPT GENERATION (Execution {execution_id}) ===")
        print(f"Target: {num_prompts} prompts")

        # Load the prompt files
        self.load_prompts(system_prompt_file, project_info_file)

        # Reset state for this execution
        self.prompts = []
        self.classes = []
        self.conversation_history = []

        self._start_ollama_server()

        # Step 1: Initialize conversation with system prompt and project info
        initial_prompt = (
            f"{self.combine_prompts()}\n\n"
            f"I need you to generate {num_prompts} image prompts for this project. "
            f"Please give me the first prompt. Format your response with the prompt inside <prompt></prompt> tags."
        )

        # Start conversation
        self.conversation_history = [
            {'role': 'user', 'content': initial_prompt}
        ]

        # Generate prompts one by one
        for i in range(num_prompts):
            print(f"\n--- Requesting prompt {i + 1}/{num_prompts} ---")

            # Send conversation to model
            response = self.send_conversation_to_model(self.conversation_history)
            response_text = response['message']['content']

            # Add model response to conversation history
            self.conversation_history.append({
                'role': 'assistant',
                'content': response_text
            })

            # Extract prompt from response
            prompt = self.parse_single_prompt(response_text)

            if prompt:
                self.prompts.append(prompt)
                print(f"✓ Prompt {i + 1}: {prompt[:80]}...")
            else:
                print(f"✗ Failed to extract prompt {i + 1}")

            # Prepare next request (except for the last iteration)
            if i < num_prompts - 1:
                next_request = "Give me another prompt. Format your response with the prompt inside <prompt></prompt> tags."
                self.conversation_history.append({
                    'role': 'user',
                    'content': next_request
                })

        # Step 2: Ask for classes
        if classes:
            print(f"\n--- Requesting classes ---")
            classes_request = (
                f"Now give me the classes for all the prompts we've discussed. "
                f"Format your response with the classes inside <classes></classes> tags, "
                f"separated by periods (e.g., <classes>class1.class2.class3</classes>). Remember to"
                f"follow project info and generate simple words (max. in project info)."
            )

            self.conversation_history.append({
                'role': 'user',
                'content': classes_request
            })

            response = self.send_conversation_to_model(self.conversation_history)
            response_text = response['message']['content']

            # Extract classes
            self.classes = self.parse_classes(response_text)

        print(f"\n=== EXECUTION {execution_id} COMPLETE ===")
        print(f"Generated: {len(self.prompts)} prompts, {len(self.classes)} classes")

        return self.prompts, self.classes

    def save_results(self,
                     output_dir: str = "output",
                     prompts_filename: str = "prompts.txt",
                     classes_filename: str = "classes.txt",
                     raw_response_filename: str = "raw_response.txt") -> Dict[str, str]:
        """
        Save the extracted prompts, classes, and raw response to files.

        Args:
            output_dir: Directory to save files
            prompts_filename: Filename for prompts
            classes_filename: Filename for classes
            raw_response_filename: Filename for raw response

        Returns:
            Dictionary with saved file paths
        """
        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        filepaths = {}

        # Save prompts
        if self.prompts:
            prompts_path = os.path.join(output_dir, prompts_filename)
            with open(prompts_path, 'w', encoding='utf-8') as f:
                for i, prompt in enumerate(self.prompts, 1):
                    f.write(f"{i}. {prompt}\n\n")
            filepaths['prompts'] = prompts_path
            print(f"Prompts saved to: {prompts_path}")

        # Save classes
        if self.classes:
            classes_path = os.path.join(output_dir, classes_filename)
            with open(classes_path, 'w', encoding='utf-8') as f:
                f.write('.'.join(self.classes))
            filepaths['classes'] = classes_path
            print(f"Classes saved to: {classes_path}")

        # Save raw response (last response or conversation history)
        if self.conversation_history:
            # Save conversation history for iterative mode
            response_path = os.path.join(output_dir, "conversation_history.txt")
            with open(response_path, 'w', encoding='utf-8') as f:
                for i, msg in enumerate(self.conversation_history):
                    f.write(f"=== Message {i + 1} ({msg['role']}) ===\n")
                    f.write(f"{msg['content']}\n\n")
            filepaths['conversation'] = response_path
            print(f"Conversation history saved to: {response_path}")
        elif self.last_response:
            # Save single response for batch mode
            response_path = os.path.join(output_dir, raw_response_filename)
            with open(response_path, 'w', encoding='utf-8') as f:
                f.write(self.last_response)
            filepaths['raw_response'] = response_path
            print(f"Raw response saved to: {response_path}")

        return filepaths

    def get_summary(self) -> Dict:
        """
        Get a summary of the current state and results.

        Returns:
            Dictionary with summary information
        """
        return {
            'model_name': self.model_name,
            'system_prompt_loaded': bool(self.system_prompt),
            'project_info_loaded': bool(self.project_info),
            'response_received': bool(self.last_response or self.conversation_history),
            'prompts_count': len(self.prompts),
            'classes_count': len(self.classes),
            'prompts': self.prompts,
            'classes': self.classes,
            'mode': 'iterative' if self.conversation_history else 'batch'
        }

    def print_results(self) -> None:
        """Print the extracted prompts and classes in a formatted way."""
        print("\n" + "=" * 80)
        print("EXTRACTION RESULTS")
        print("=" * 80)

        print(f"\nEXTRACTED PROMPTS ({len(self.prompts)}):")
        print("-" * 40)
        for i, prompt in enumerate(self.prompts, 1):
            print(f"{i}. {prompt}\n")

        print(f"EXTRACTED CLASSES ({len(self.classes)}):")
        print("-" * 40)
        print('.'.join(self.classes))
        print("\n" + "=" * 80)


# Example usage
if __name__ == '__main__':
    # Initialize the generator
    generator = PromptGenerator(model_name='deepseek-r1:latest')

    # Example file paths - adjust these to your actual files
    system_prompt_file = "system_prompt.txt"
    project_info_file = "project_info.txt"

    try:
        # Test iterative generation
        prompts, classes = generator.generate_prompts_iteratively(
            system_prompt_file,
            project_info_file,
            num_prompts=5,
            execution_id=1
        )

        # Print results
        generator.print_results()

        # Save results to files
        saved_files = generator.save_results()

        # Print summary
        summary = generator.get_summary()
        print(
            f"\nSummary: Generated {summary['prompts_count']} prompts and {summary['classes_count']} classes using {summary['mode']} mode")

    except Exception as e:
        print(f"Error: {str(e)}")