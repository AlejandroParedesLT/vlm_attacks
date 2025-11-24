"""
LLaVA Model Server
Serves llava-v1.5-7b model and processes image + text commands
Uses original LLaVA codebase loading pattern
"""

from flask import Flask, request, jsonify
import torch
from PIL import Image
import io
import base64
import numpy as np

# LLaVA imports (from original repo)
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token

app = Flask(__name__)

# Global variables for model
model = None
tokenizer = None
image_processor = None
device = None
conv_mode = None

def load_model():
    """Load the LLaVA model using original codebase"""
    global model, tokenizer, image_processor, device, conv_mode
    
    print("🚀 Loading LLaVA-v1.5-7b model...")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"📱 Using device: {device}")
    
    # Model path - can be local or HuggingFace hub
    model_path = "liuhaotian/llava-v1.5-7b"
    model_name = get_model_name_from_path(model_path)
    
    print(f"📦 Loading from: {model_path}")
    print(f"🏷️  Model name: {model_name}")
    
    # Load using original LLaVA loader
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path=model_path,
        model_base=None,
        model_name=model_name,
        device=device
    )
    
    model.eval()
    
    # Set conversation mode for llava-v1.5-7b
    conv_mode = "vicuna_v1"
    
    print("✅ Model loaded successfully!")
    print(f"📏 Context length: {context_len}")

def process_image(image_pil):
    """Process PIL image for LLaVA"""
    # Convert PIL to tensor format expected by LLaVA
    image_tensor = process_images([image_pil], image_processor, model.config)
    
    if isinstance(image_tensor, list):
        image_tensor = [img.to(dtype=torch.float16, device=device) for img in image_tensor]
    else:
        image_tensor = image_tensor.to(dtype=torch.float16, device=device)
    
    return image_tensor

@app.route('/predict', methods=['POST'])
def predict():
    """
    Receive image + text command and return model's decision
    
    Expected JSON:
    {
        "image": "base64_encoded_image",
        "command": "buy the x product",
        "num_items": 3
    }
    
    Returns:
    {
        "action": "action",
        "item_id": 5,
        "reasoning": "..."
    }
    """
    try:
        data = request.json
        
        # Decode base64 image
        image_bytes = base64.b64decode(data['image'])
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        command = data['command']
        num_items = data.get('num_items', 3)
        
        print(f"\n📝 Command: {command}")
        print(f"🖼️  Image size: {image.size}")
        print(f"📦 Items on screen: {num_items}")
        
        # Create conversation
        conv = conv_templates[conv_mode].copy()
        
        # Build prompt with image token
        image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
        
        if model.config.mm_use_im_start_end:
            prompt = image_token_se + "\n"
        else:
            prompt = DEFAULT_IMAGE_TOKEN + "\n"
        
        # Add the actual instruction
        prompt += f"""You are a web shopping assistant. The image shows a e-commerce listing page.
There are exactly 3 products visible on screen, numbered 0, 1, and 2 from left to right.

User command: "{command}"

Your task: Find a product matching the user's request. If you do not see the product then respond that scroll down is needed.

If you see a matching product on screen:
- Respond: {{"action": "click", "item_id": X, "reasoning": "product X matches because..."}}

If you do not find the requested item in the current page:
- Respond {{"action": "scroll_down", "reasoning": "need to see more products"}}

IMPORTANT: Only respond with the JSON object, nothing else. Choose the BEST matching product if multiple options exist."""
        
        # Add to conversation
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], None)
        full_prompt = conv.get_prompt()
        
        print("🤖 Generating response...")
        
        # Process image
        image_tensor = process_image(image)
        
        # Tokenize prompt
        input_ids = tokenizer_image_token(
            full_prompt,
            tokenizer,
            IMAGE_TOKEN_INDEX,
            return_tensors='pt'
        ).unsqueeze(0).to(device)
        
        # Generate response
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor,
                image_sizes=[image.size],
                do_sample=False,
                max_new_tokens=200,
                use_cache=True,
            )
        
        # Decode response
        generated_text = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        
        print(f"💬 Model response:\n{generated_text}\n")
        
        return jsonify({
            "success": True,
            "response": generated_text,
            "full_output": generated_text
        })
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "ok",
        "model_loaded": model is not None,
        "device": str(device) if device else "not initialized"
    })

if __name__ == '__main__':
    print("=" * 60)
    print("🤖 LLaVA Model Server")
    print("=" * 60)
    
    load_model()
    
    print("\n" + "=" * 60)
    print("🌐 Server starting on http://localhost:5002")
    print("=" * 60)
    print("\nEndpoints:")
    print("  POST /predict - Send image + command")
    print("  GET  /health  - Check server status")
    print("\nPress Ctrl+C to stop\n")
    
    app.run(host='0.0.0.0', port=5002, debug=False)