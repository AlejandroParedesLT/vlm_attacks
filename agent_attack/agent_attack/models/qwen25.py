"""
qwen25_vlm.py

Class definition for Qwen2.5-VL, wrapping utilities for VQA, image captioning, 
conditional likelihood estimation, and forward pass with loss computation.

Plug-and-play replacement for LLaVA VLM class.
"""
from pathlib import Path
from typing import Callable, List, Optional, Tuple, Union,Any

import numpy as np
import torch
import torch.nn as nn
from accelerate import PartialState
from PIL import Image
from transformers import AutoProcessor
from torchvision.transforms import Compose, InterpolationMode, Normalize, Resize

try:
    from transformers import Qwen2_5_VLForConditionalGeneration
    QWEN_25_AVAILABLE = True
except ImportError:
    QWEN_25_AVAILABLE = False
    raise ImportError("Qwen2.5-VL not available. Install: pip install transformers>=4.37.0")

from qwen_vl_utils import process_vision_info

from agent_attack.util.interfaces import VLM


class Qwen25VL(VLM):
    def __init__(
        self,
        hub_path: str,
        max_length: int = 512,
        temperature: float = 0.2,
        **_: str,
    ) -> None:
        self.hub_path = hub_path
        self.dtype = torch.bfloat16
        self.model_id = self.hub_path.split("/")[-1]

        # Get Distributed State
        self.distributed_state = PartialState()

        # Load Model on GPU(s)
        self.model, self.processor, self.image_processor_raw = self.load()

        # Setup image processor from tensor (for adversarial attacks and tensor inputs)
        # Qwen2.5-VL uses patch size of 14, so ensure dimensions are divisible by 14
        if hasattr(self.image_processor_raw, 'size'):
            if isinstance(self.image_processor_raw.size, dict):
                h = self.image_processor_raw.size.get('height', 448)
                w = self.image_processor_raw.size.get('width', 448)
            else:
                h = w = self.image_processor_raw.size
        else:
            # Default size for Qwen2.5-VL - must be divisible by 14
            h = w = 448

        # Ensure dimensions are divisible by 14 (patch size)
        h = ((h + 13) // 14) * 14
        w = ((w + 13) // 14) * 14
        resize = (h, w)

        # Get normalization stats
        if hasattr(self.image_processor_raw, 'image_mean'):
            mean = self.image_processor_raw.image_mean
            std = self.image_processor_raw.image_std
        else:
            # Default ImageNet stats
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]

        self.image_processor_from_tensor = Compose(
            [
                Resize(resize, interpolation=InterpolationMode.BICUBIC, antialias=True),
                Normalize(mean=mean, std=std),
            ]
        )

        # Set Default Generation Configuration
        self.max_length = max_length
        self.temperature = temperature
        self.generate_kwargs = {
            "do_sample": False if temperature == 0.0 else True,
            "max_new_tokens": self.max_length,
            "temperature": self.temperature if temperature > 0.0 else None,
        }

        # For computing likelihoods --> get tokens corresponding to "True", "False", "Yes", "No"
        self.string2idx = {}
        for trigger_string in ["True", "False", "Yes", "No"] + [chr(ord("A") + i) for i in range(26)]:
            token_idx_list = self.processor.tokenizer.encode(trigger_string, add_special_tokens=False)
            assert len(token_idx_list) == 1, f'String "{trigger_string}" is tokenized as more than one token!'
            self.string2idx[trigger_string] = token_idx_list[0]

    def load(self) -> Tuple[nn.Module, AutoProcessor, Any]:
        """Loads Qwen2.5-VL model and processor"""
        print(f"Loading Qwen2.5-VL model from {self.hub_path}...")
        
        with self.distributed_state.main_process_first():
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.hub_path,
                torch_dtype=self.dtype,
                device_map="auto",
            )
            processor = AutoProcessor.from_pretrained(self.hub_path)

        model.eval()
        
        # Extract image processor for compatibility
        image_processor_raw = processor.image_processor
        
        return model, processor, image_processor_raw

    def freeze(self) -> None:
        """Freeze all model parameters"""
        for param in self.model.parameters():
            param.requires_grad = False

    def set_generate_kwargs(self, generate_kwargs):
        """Update generation kwargs"""
        self.generate_kwargs = generate_kwargs

    def get_captioning_prompt_fn(self) -> Callable[[], str]:
        """Generates the full reference prompt for captioning tasks."""
        def qwen_cap_prompt_fn() -> str:
            return "Provide a short image description."
        
        return qwen_cap_prompt_fn

    def get_agent_prompt_fn(self) -> Callable[[str], str]:
        """Generates the full reference prompt for agent tasks."""
        def qwen_agent_prompt_fn(agent_prompt: str) -> str:
            return agent_prompt
        
        return qwen_agent_prompt_fn

    def _prepare_messages(self, image: Image.Image, text: str) -> List[dict]:
        """Helper to prepare messages in Qwen chat format"""
        return [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": text},
                ],
            }
        ]

    def _process_inputs(self, images: List[Image.Image], texts: List[str]):
        """Process images and texts into model inputs"""
        all_messages = [self._prepare_messages(img, txt) for img, txt in zip(images, texts)]
        
        # Apply chat template to all messages
        processed_texts = [
            self.processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            for msgs in all_messages
        ]
        
        # Process vision info for all messages
        all_image_inputs = []
        all_video_inputs = []
        for msgs in all_messages:
            image_inputs, video_inputs = process_vision_info(msgs)
            all_image_inputs.extend(image_inputs if image_inputs else [])
            all_video_inputs.extend(video_inputs if video_inputs else [])
        
        # Prepare batch inputs
        inputs = self.processor(
            text=processed_texts,
            images=all_image_inputs if all_image_inputs else None,
            videos=all_video_inputs if all_video_inputs else None,
            padding=True,
            return_tensors="pt",
        )
        
        # Move to device
        inputs = {
            k: v.to(self.distributed_state.device) if isinstance(v, torch.Tensor) else v 
            for k, v in inputs.items()
        }
        
        return inputs

    @torch.inference_mode()
    def generate_answer(
        self,
        pixel_values: Union[torch.Tensor, List[Image.Image]],
        questions: List[str],
        return_string_probabilities: Optional[List[str]] = None,
        image_sizes: Optional[torch.LongTensor] = None,
        temperature: Optional[float] = None,
    ) -> Union[List[str], List[List[float]]]:
        """
        Generate answers for given images and questions.
        
        Args:
            pixel_values: Either torch.Tensor or List[Image.Image] (for compatibility)
            questions: List of text prompts/questions
            return_string_probabilities: Optional list of strings to compute probabilities for
            image_sizes: Not used for Qwen (kept for API compatibility)
            temperature: Optional temperature override
            
        Returns:
            List of generated text responses, or list of probability distributions
        """
        # Update generation kwargs with temperature if provided
        generate_kwargs = self.generate_kwargs.copy()
        if temperature is not None and temperature > 0:
            generate_kwargs["temperature"] = temperature
            generate_kwargs["do_sample"] = True

        # Convert pixel_values to images if needed (for LLaVA compatibility)
        if isinstance(pixel_values, torch.Tensor):
            # Convert tensor back to PIL Images
            images = []
            for i in range(pixel_values.shape[0]):
                # Denormalize and convert to PIL
                img_tensor = pixel_values[i]
                # Assuming standard ImageNet normalization
                mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(img_tensor.device)
                std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(img_tensor.device)
                img_tensor = img_tensor * std + mean
                img_tensor = torch.clamp(img_tensor * 255, 0, 255).byte()
                img_np = img_tensor.permute(1, 2, 0).cpu().numpy()
                images.append(Image.fromarray(img_np))
        else:
            images = pixel_values

        gen_texts, gen_probabilities = [], []

        # Process each image-question pair
        for idx, (image, question) in enumerate(zip(images, questions)):
            messages = self._prepare_messages(image, question)
            
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
            inputs = {
                k: v.to(self.distributed_state.device) if isinstance(v, torch.Tensor) else v 
                for k, v in inputs.items()
            }

            if return_string_probabilities is None:
                # Standard generation
                with torch.cuda.amp.autocast(dtype=self.dtype):
                    generated_ids = self.model.generate(**inputs, **generate_kwargs)
                
                # Trim input tokens from output
                generated_ids_trimmed = [
                    out_ids[len(in_ids):] 
                    for in_ids, out_ids in zip(inputs['input_ids'], generated_ids)
                ]
                
                # Decode
                generated_text = self.processor.batch_decode(
                    generated_ids_trimmed,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )[0].strip()
                
                gen_texts.append(generated_text)
            
            else:
                # Generation with probability computation
                with torch.amp.autocast('cuda', dtype=self.dtype):
                    output_dict = self.model.generate(
                        **inputs,
                        output_scores=True,
                        return_dict_in_generate=True,
                        **generate_kwargs,
                    )
                
                # Decode generated text
                generated_ids_trimmed = [
                    out_ids[len(in_ids):] 
                    for in_ids, out_ids in zip(inputs['input_ids'], output_dict.sequences)
                ]
                generated_text = self.processor.batch_decode(
                    generated_ids_trimmed,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )[0].strip()
                gen_texts.append(generated_text)
                
                # Get token probabilities from first generated token
                token_probs = torch.softmax(output_dict.scores[0][0], dim=0)
                
                # Get normalized probabilities for specified strings
                slice_idxs = torch.tensor([self.string2idx[s] for s in return_string_probabilities])
                string_probs_unnormalized = token_probs[slice_idxs]
                string_probs = string_probs_unnormalized / string_probs_unnormalized.sum()
                gen_probabilities.append(string_probs.cpu().numpy().tolist())

        return gen_texts if return_string_probabilities is None else gen_probabilities

    def forward(
        self,
        pixel_values: Union[torch.Tensor, List[Image.Image]],
        questions: List[str],
        answers: List[str],
        image_sizes: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass through the model, computing the loss for the given inputs.
        
        Args:
            pixel_values: Either torch.Tensor or List[Image.Image]
            questions: List of question prompts
            answers: List of target answers
            image_sizes: Not used for Qwen (kept for API compatibility)
            
        Returns:
            Loss tensor
        """
        # For tensor inputs (adversarial attacks), we need to process differently
        # to maintain gradient flow
        if isinstance(pixel_values, torch.Tensor):
            return self._forward_from_tensor(pixel_values, questions, answers)
        else:
            return self._forward_from_images(pixel_values, questions, answers)

    def _forward_from_images(
        self,
        images: List[Image.Image],
        questions: List[str],
        answers: List[str],
    ) -> torch.Tensor:
        """Forward pass when inputs are PIL Images (standard case)"""
        total_loss = 0.0

        # Process each example individually
        for idx, (image, question, answer) in enumerate(zip(images, questions, answers)):
            # Prepare messages for prompt (question only)
            prompt_messages = self._prepare_messages(image, question)
            
            # Prepare messages for full sequence (question + answer)
            full_text = question + " " + answer
            full_messages = self._prepare_messages(image, full_text)
            
            # Process prompt to get prompt length
            prompt_text = self.processor.apply_chat_template(
                prompt_messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(prompt_messages)
            prompt_inputs = self.processor(
                text=[prompt_text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            prompt_length = prompt_inputs['input_ids'].shape[1]
            
            # Process full sequence
            full_text_formatted = self.processor.apply_chat_template(
                full_messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(full_messages)
            full_inputs = self.processor(
                text=[full_text_formatted],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            
            # Move to device
            full_inputs = {
                k: v.to(self.distributed_state.device) if isinstance(v, torch.Tensor) else v 
                for k, v in full_inputs.items()
            }
            
            # Add EOS token
            eos_token_id = self.processor.tokenizer.eos_token_id
            input_ids_with_eos = torch.cat([
                full_inputs['input_ids'],
                torch.tensor([[eos_token_id]], device=self.distributed_state.device)
            ], dim=1)
            
            # Create labels: -100 for prompt tokens, actual tokens for answer
            labels = input_ids_with_eos.clone()
            labels[0, :prompt_length] = -100
            
            # Adjust attention mask for EOS token
            attention_mask = torch.cat([
                full_inputs['attention_mask'],
                torch.ones((1, 1), device=self.distributed_state.device, dtype=full_inputs['attention_mask'].dtype)
            ], dim=1)
            
            # Forward pass with loss computation
            with torch.amp.autocast('cuda', dtype=self.dtype):
                outputs = self.model(
                    input_ids=input_ids_with_eos,
                    attention_mask=attention_mask,
                    pixel_values=full_inputs.get('pixel_values'),
                    image_grid_thw=full_inputs.get('image_grid_thw'),
                    labels=labels,
                )
            
            total_loss += outputs.loss

        return total_loss

    def _forward_from_tensor(
        self,
        pixel_values: torch.Tensor,
        questions: List[str],
        answers: List[str],
    ) -> torch.Tensor:
        """
        Forward pass when inputs are tensors (for adversarial attacks).
        This version maintains gradient flow by working directly with tensors.
        """
        total_loss = 0.0
        
        # Process each example
        for idx in range(pixel_values.shape[0]):
            question = questions[idx]
            answer = answers[idx]
            
            # Get the pixel tensor for this example
            img_tensor = pixel_values[idx:idx+1]  # Keep batch dimension
            
            # Prepare text inputs
            full_text = question + " " + answer
            
            # Create dummy messages just to get the text template
            # We'll replace the actual image tensors later
            dummy_image = Image.new('RGB', (224, 224))
            prompt_messages = self._prepare_messages(dummy_image, question)
            full_messages = self._prepare_messages(dummy_image, full_text)
            
            # Get prompt length (text only, no actual image processing)
            prompt_text = self.processor.apply_chat_template(
                prompt_messages, tokenize=False, add_generation_prompt=True
            )
            prompt_tokens = self.processor.tokenizer(
                prompt_text,
                return_tensors="pt",
                padding=True,
            )
            prompt_length = prompt_tokens['input_ids'].shape[1]
            
            # Process full text
            full_text_formatted = self.processor.apply_chat_template(
                full_messages, tokenize=False, add_generation_prompt=True
            )
            text_inputs = self.processor.tokenizer(
                full_text_formatted,
                return_tensors="pt",
                padding=True,
            )
            
            # Move to device
            input_ids = text_inputs['input_ids'].to(self.distributed_state.device)
            attention_mask = text_inputs['attention_mask'].to(self.distributed_state.device)
            
            # Add EOS token
            eos_token_id = self.processor.tokenizer.eos_token_id
            input_ids_with_eos = torch.cat([
                input_ids,
                torch.tensor([[eos_token_id]], device=self.distributed_state.device)
            ], dim=1)
            
            # Create labels
            labels = input_ids_with_eos.clone()
            labels[0, :prompt_length] = -100
            
            # Adjust attention mask
            attention_mask = torch.cat([
                attention_mask,
                torch.ones((1, 1), device=self.distributed_state.device, dtype=attention_mask.dtype)
            ], dim=1)
            
            # For Qwen, we need to process the image tensor into the format it expects
            # The pixel_values here are already normalized (from image_processor_from_tensor)
            # We need to prepare them in Qwen's expected format
            
            # Qwen expects specific image grid info - we'll use default values
            # This is a simplified version - may need adjustment based on actual image sizes
            image_grid_thw = torch.tensor([[1, img_tensor.shape[2] // 14, img_tensor.shape[3] // 14]], 
                                          device=self.distributed_state.device)
            
            # Forward pass with gradient tracking enabled
            with torch.amp.autocast('cuda', dtype=self.dtype):
                outputs = self.model(
                    input_ids=input_ids_with_eos,
                    attention_mask=attention_mask,
                    pixel_values=img_tensor,
                    image_grid_thw=image_grid_thw,
                    labels=labels,
                )
            
            total_loss += outputs.loss

        return total_loss

    def image_processor(self, images, return_tensors="pt"):
        """
        Process images - compatible with LLaVA API.
        Converts PIL images or list of images to tensors with proper preprocessing.
        """
        if return_tensors == "pt":
            if isinstance(images, Image.Image):
                images = [images]
            
            # Convert PIL images to tensors with normalization
            # This matches the LLaVA preprocessing pipeline
            images_tensors = torch.cat(
                [
                    torch.from_numpy(np.array(image).astype(np.float32) / 255.0)
                    .unsqueeze(0)
                    .permute(0, 3, 1, 2)
                    for image in images
                ],
                dim=0,
            )
            
            # Apply the image_processor_from_tensor transform
            pixel_values = self.image_processor_from_tensor(images_tensors)
            
            return {"pixel_values": pixel_values}
        else:
            raise NotImplementedError("Only return_tensors='pt' is supported!")


if __name__ == "__main__":
    # Example usage
    model = Qwen25VL("Qwen/Qwen2.5-VL-7B-Instruct")
    
    # Test captioning
    from PIL import Image
    image = Image.open("test.jpeg").convert("RGB")
    prompt_fn = model.get_captioning_prompt_fn()
    response = model.generate_answer([image], [prompt_fn()])
    print("Caption:", response)
    
    # Test with probabilities
    question = "Is this a cat? Answer with True or False."
    probs = model.generate_answer(
        [image], 
        [question], 
        return_string_probabilities=["True", "False"]
    )
    print("Probabilities:", probs)
    
    # Test forward pass (loss computation)
    loss = model.forward([image], [question], ["True"])
    print("Loss:", loss.item())