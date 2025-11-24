"""
qwen.py

Class definition for wrapping Qwen2.5-VL, wrapping
utilities for VQA, image captioning, and conditional likelihood estimation.
"""
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from accelerate import PartialState
from PIL import Image
from transformers import AutoProcessor
try:
    from transformers import Qwen2_5_VLForConditionalGeneration
    QWEN_25_AVAILABLE = True
except ImportError:
    QWEN_25_AVAILABLE = False
    
# try:
#     from transformers import Qwen2VLForConditionalGeneration
#     QWEN_2_AVAILABLE = True
# except ImportError:
#     QWEN_2_AVAILABLE = False

from qwen_vl_utils import process_vision_info

from agent_attack.util.interfaces import VLM


class Qwen(VLM):
    def __init__(
        self,
        hub_path: str,
        max_length: int = 512,
        temperature: float = 0.0,
        **_: str,
    ) -> None:
        self.hub_path = hub_path

        # Get Distributed State
        self.distributed_state = PartialState()

        # Load model and processor
        print(f"Loading Qwen model from {hub_path}...")
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            hub_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        self.processor = AutoProcessor.from_pretrained(hub_path)
        
        self.model.eval()

        # Set Default VQA Generation Configuration
        self.max_length = max_length
        self.temperature = temperature
        self.generate_kwargs = {
            "max_new_tokens": self.max_length,
            "do_sample": False if temperature == 0.0 else True,
            "temperature": temperature if temperature > 0.0 else None,
        }

    def set_generate_kwargs(self, generate_kwargs):
        self.generate_kwargs = generate_kwargs

    def get_prompt_builder(self, system_prompt: Optional[str] = None) -> Any:
        """Returns a prompt builder for Qwen's chat format"""
        return lambda text: [{"role": "user", "content": [{"type": "text", "text": text}]}]

    def get_captioning_prompt_fn(self) -> Callable[[str], str]:
        """Generates the full reference prompt for captioning tasks."""

        def captioning_prompt_fn() -> str:
            cap_prompt = "Provide a short image description in one sentence (e.g., the color, the object, the activity, and the location)."
            return cap_prompt

        return captioning_prompt_fn

    @torch.inference_mode()
    def generate_answer(
        self,
        images: List[Image.Image],
        question_prompts: List[str],
        return_string_probabilities: Optional[List[str]] = None,
        image_sizes: Optional[torch.LongTensor] = None,
        temperature: Optional[float] = None,
    ) -> Union[List[str], List[Tuple[str, List[float]]]]:
        """
        Generate answers for given images and prompts.
        
        Args:
            images: List of PIL Images
            question_prompts: List of text prompts
            return_string_probabilities: Optional list of strings to compute probabilities for
            image_sizes: Not used for Qwen
            temperature: Optional temperature override
            
        Returns:
            List of generated text responses, or tuples of (text, probabilities) if return_string_probabilities is provided
        """
        # Update generation kwargs with temperature if provided
        generate_kwargs = self.generate_kwargs.copy()
        if temperature is not None:
            generate_kwargs["temperature"] = temperature if temperature > 0.0 else None
            generate_kwargs["do_sample"] = temperature > 0.0

        if return_string_probabilities is not None:
            # For probability computation, we need to handle this differently
            return self._generate_with_probabilities(
                images, question_prompts, return_string_probabilities, generate_kwargs
            )

        responses = []
        for image, question_prompt in zip(images, question_prompts, strict=True):
            # Create message in Qwen's chat format
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": question_prompt},
                    ],
                }
            ]

            # Apply chat template
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            # Process vision info
            image_inputs, video_inputs = process_vision_info(messages)

            # Prepare inputs
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to(self.distributed_state.device)

            # Generate
            generated_ids = self.model.generate(**inputs, **generate_kwargs)

            # Trim input tokens from output
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] 
                for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]

            # Decode
            response = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]

            print("output:", response)
            responses.append(response)

        return responses

    def _generate_with_probabilities(
        self,
        images: List[Image.Image],
        question_prompts: List[str],
        candidate_strings: List[str],
        generate_kwargs: dict,
    ) -> List[Tuple[str, List[float]]]:
        """
        Generate answers and compute probabilities for candidate strings.
        
        This method computes the log-likelihood of each candidate string given the image and prompt.
        """
        results = []
        
        for image, question_prompt in zip(images, question_prompts, strict=True):
            # Create message in Qwen's chat format
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": question_prompt},
                    ],
                }
            ]

            # Apply chat template
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            # Process vision info
            image_inputs, video_inputs = process_vision_info(messages)

            # Prepare inputs
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to(self.distributed_state.device)

            # Compute probabilities for each candidate
            probabilities = []
            for candidate in candidate_strings:
                # Tokenize the candidate answer
                candidate_tokens = self.processor.tokenizer.encode(
                    candidate, add_special_tokens=False, return_tensors="pt"
                ).to(self.distributed_state.device)

                # Prepare full sequence (input + candidate)
                full_input_ids = torch.cat([inputs.input_ids, candidate_tokens], dim=1)

                # Get logits
                with torch.no_grad():
                    outputs = self.model(
                        input_ids=full_input_ids,
                        attention_mask=torch.ones_like(full_input_ids),
                        pixel_values=inputs.get("pixel_values"),
                        image_grid_thw=inputs.get("image_grid_thw"),
                    )
                    logits = outputs.logits

                # Compute log probability of the candidate tokens
                log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
                
                # Get log probs for candidate tokens
                candidate_log_probs = []
                for i, token_id in enumerate(candidate_tokens[0]):
                    token_log_prob = log_probs[0, len(inputs.input_ids[0]) + i - 1, token_id]
                    candidate_log_probs.append(token_log_prob.item())
                
                # Sum log probs (or average)
                total_log_prob = sum(candidate_log_probs)
                probabilities.append(total_log_prob)

            # Generate actual response
            generated_ids = self.model.generate(**inputs, **generate_kwargs)
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] 
                for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            response = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]

            results.append((response, probabilities))

        return results


if __name__ == "__main__":
    # Example usage
    model = Qwen("Qwen/Qwen2.5-VL-7B-Instruct")
    prompt_fn = model.get_captioning_prompt_fn()
    image = Image.open("test.jpeg").convert("RGB")
    response = model.generate_answer([image], [prompt_fn()])
    print(response)