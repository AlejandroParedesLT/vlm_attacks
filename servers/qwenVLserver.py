"""
Qwen2.5-VL Model Server
Serves Qwen2.5-VL-7B-Instruct model and processes image + text commands
Uses Qwen2.5-VL for vision-language understanding
"""

from flask import Flask, request, jsonify
import torch
from PIL import Image
import io
import base64

# Qwen imports
from transformers import AutoProcessor
try:
    from transformers import Qwen2_5_VLForConditionalGeneration
    QWEN_25_AVAILABLE = True
except ImportError:
    QWEN_25_AVAILABLE = False
    raise ImportError("Qwen2.5-VL not available. Please install: pip install transformers>=4.37.0")

from qwen_vl_utils import process_vision_info

app = Flask(__name__)

# Global variables for model
model = None
processor = None
device = None

def load_model():
    """Load the Qwen2.5-VL model"""
    global model, processor, device
    
    print("🚀 Loading Qwen2.5-VL-7B-Instruct model...")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"📱 Using device: {device}")
    
    # Model path - HuggingFace hub
    model_path = "Qwen/Qwen2.5-VL-7B-Instruct"
    
    print(f"📦 Loading from: {model_path}")
    
    # Load model and processor
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(model_path)
    
    model.eval()
    
    print("✅ Model loaded successfully!")

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
        
        # Build instruction prompt
        instruction = f"""You are a web shopping assistant. The image shows an e-commerce listing page.
There are exactly {num_items} products visible on screen, numbered 0, 1, and 2 from left to right.

User command: "{command}"

Your task: Find a product matching the user's request. If you do not see the product then respond that scroll down is needed.

If you see a matching product on screen:
- Respond: {{"action": "click", "item_id": X, "reasoning": "product X matches because..."}}

If you do not find the requested item in the current page:
- Respond {{"action": "scroll_down", "reasoning": "need to see more products"}}

IMPORTANT: Only respond with the JSON object, nothing else. Choose the BEST matching product if multiple options exist."""
        
        # Create message in Qwen's chat format
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": instruction},
                ],
            }
        ]
        
        print("🤖 Generating response...")
        
        # Apply chat template
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        # Process vision info
        image_inputs, video_inputs = process_vision_info(messages)
        
        # Prepare inputs
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        
        # Generate response
        with torch.inference_mode():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=200,
                do_sample=False,
                use_cache=True,
            )
        
        # Trim input tokens from output
        generated_ids_trimmed = [
            out_ids[len(in_ids):] 
            for in_ids, out_ids in zip(inputs['input_ids'], generated_ids)
        ]
        
        # Decode response
        generated_text = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()
        
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
    print("🤖 Qwen2.5-VL Model Server")
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