"""
LLaVA Web Agent Client
Captures screenshots, sends to LLaVA, and executes actions (scroll/click)
"""

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from PIL import Image, ImageDraw, ImageFont
import io
import base64
import requests
import time
import json
import re

class WebAgent:
    def __init__(self, url="http://localhost:5000", llava_url="http://localhost:5001"):
        self.url = url
        self.llava_url = llava_url
        self.driver = None
        self.current_scroll_position = 0
        self.items_per_view = 3  # Show only 3 items at a time
        
    def setup_browser(self):
        """Initialize Chrome browser"""
        print("🌐 Initializing browser...")
        options = webdriver.ChromeOptions()
        # Remove headless to see actions happen
        # options.add_argument('--headless')
        options.add_argument('--window-size=1920,1080')
        options.add_argument('--disable-gpu')
        self.driver = webdriver.Chrome(options=options)
        self.driver.get(self.url)
        time.sleep(2)
        print("✅ Browser initialized")
        
    def get_visible_products(self):
        """Get only the 3 visible products in the viewport"""
        all_products = self.driver.find_elements(By.CLASS_NAME, "product-card")
        
        visible_products = []
        for product in all_products:
            location = product.location
            size = product.size
            
            # Check if product is in viewport
            viewport_height = self.driver.execute_script("return window.innerHeight")
            if location['y'] >= 0 and location['y'] + size['height'] <= viewport_height + 200:
                visible_products.append(product)
                
        return visible_products[:self.items_per_view]
    
    def capture_and_annotate(self):
        """
        Capture screenshot and add numbered bounding boxes to visible products
        Returns: PIL.Image, int (number of items), list of product info
        """
        # Take screenshot
        screenshot_bytes = self.driver.get_screenshot_as_png()
        screenshot = Image.open(io.BytesIO(screenshot_bytes))
        
        # Get visible products
        visible_products = self.get_visible_products()
        
        print(f"📦 Found {len(visible_products)} visible products")
        
        # Create drawing context
        draw = ImageDraw.Draw(screenshot)
        
        # Load font
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 28)
        except:
            try:
                font = ImageFont.truetype("arial.ttf", 28)
            except:
                font = ImageFont.load_default()
        
        # Store product info
        products_info = []
        
        # Draw bounding boxes with numbers
        for idx, product in enumerate(visible_products):
            location = product.location
            size = product.size
            
            x1 = location['x']
            y1 = location['y']
            x2 = x1 + size['width']
            y2 = y1 + size['height']
            
            # Draw red bounding box
            draw.rectangle([x1, y1, x2, y2], outline="red", width=4)
            
            # Draw number label with background
            label = str(idx)
            
            try:
                bbox = draw.textbbox((x1 + 10, y1 + 10), label, font=font)
                padding = 5
                draw.rectangle([bbox[0]-padding, bbox[1]-padding, 
                              bbox[2]+padding, bbox[3]+padding], fill="red")
                draw.text((x1 + 10, y1 + 10), label, fill="white", font=font)
            except:
                draw.rectangle([x1 + 10, y1 + 10, x1 + 50, y1 + 45], fill="red")
                draw.text((x1 + 15, y1 + 15), label, fill="white", font=font)
            
            # Store product info
            try:
                title = product.find_element(By.CLASS_NAME, "product-title").text
                price = product.find_element(By.CLASS_NAME, "product-price").text
                product_id = product.get_attribute("data-product-id")
                products_info.append({
                    "id": idx,
                    "product_id": product_id,
                    "title": title,
                    "price": price,
                    "element": product
                })
            except:
                products_info.append({
                    "id": idx,
                    "element": product
                })
        
        return screenshot, len(visible_products), products_info
    
    def send_to_llava(self, image, command, num_items):
        """Send image and command to LLaVA server"""
        buffer = io.BytesIO()
        image.save(buffer, format='PNG')
        image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        payload = {
            "image": image_base64,
            "command": command,
            "num_items": num_items
        }
        
        print(f"🚀 Sending to LLaVA...")
        response = requests.post(
            f"{self.llava_url}/predict",
            json=payload,
            timeout=120
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"Server error: {response.status_code}")
    
    def parse_action(self, llava_response):
        """
        Parse LLaVA response to extract action
        Returns: dict with action details
        """
        response_text = llava_response.get("response", "")
        
        # Try to parse as JSON first
        try:
            # Find JSON in the response
            json_match = re.search(r'\{[^}]+\}', response_text)
            if json_match:
                action_data = json.loads(json_match.group())
                return action_data
        except:
            pass
        
        # Fallback: parse text
        response_lower = response_text.lower()
        
        if "click" in response_lower:
            # Try to find item number
            numbers = re.findall(r'\d+', response_text)
            if numbers:
                return {
                    "action": "click",
                    "item_id": int(numbers[0]),
                    "reasoning": response_text
                }
        elif "scroll_down" in response_lower or "scroll down" in response_lower:
            return {
                "action": "scroll_down",
                "reasoning": response_text
            }
        elif "scroll_up" in response_lower or "scroll up" in response_lower:
            return {
                "action": "scroll_up",
                "reasoning": response_text
            }
        
        return {
            "action": "unknown",
            "reasoning": response_text
        }
    
    def execute_action(self, action, products_info):
        """Execute the action returned by LLaVA"""
        action_type = action.get("action", "").lower()
        
        print(f"\n🎯 EXECUTING ACTION: {action_type}")
        print(f"💭 Reasoning: {action.get('reasoning', 'N/A')}")
        
        if action_type == "click":
            item_id = action.get("item_id")
            if item_id is not None and 0 <= item_id < len(products_info):
                product = products_info[item_id]
                print(f"🖱️  Clicking on item {item_id}: {product.get('title', 'Unknown')}")
                
                # Click the product
                try:
                    product['element'].click()
                    time.sleep(2)
                    print("✅ Click executed - Modal should appear!")
                    return True
                except Exception as e:
                    print(f"❌ Click failed: {e}")
                    return False
            else:
                print(f"❌ Invalid item_id: {item_id}")
                return False
                
        elif action_type == "scroll_down":
            print("⬇️  Scrolling down (showing next 3 items)...")
            # Scroll by approximately 3 products
            self.driver.execute_script("window.scrollBy(0, 800);")
            time.sleep(1)
            self.current_scroll_position += 1
            print(f"✅ Scrolled down (position: {self.current_scroll_position})")
            return True
            
        elif action_type == "scroll_up":
            print("⬆️  Scrolling up (showing previous 3 items)...")
            # Scroll up by approximately 3 products
            self.driver.execute_script("window.scrollBy(0, -800);")
            time.sleep(1)
            self.current_scroll_position = max(0, self.current_scroll_position - 1)
            print(f"✅ Scrolled up (position: {self.current_scroll_position})")
            return True
            
        else:
            print(f"❌ Unknown action: {action_type}")
            return False
    
    def run(self, command, max_iterations=5):
        """
        Main loop: capture -> send to LLaVA -> execute action
        Continues until item is clicked or max iterations reached
        """
        print("=" * 70)
        print("🤖 WEB AGENT WITH VISION")
        print("=" * 70)
        print(f"\n💬 Command: {command}")
        print(f"🔄 Max iterations: {max_iterations}")
        print("\n" + "-" * 70)
        
        self.setup_browser()
        
        iteration = 0
        action_successful = False
        
        while iteration < max_iterations and not action_successful:
            iteration += 1
            print(f"\n{'='*70}")
            print(f"ITERATION {iteration}/{max_iterations}")
            print(f"{'='*70}")
            
            # Step 1: Capture and annotate
            print("\n📸 Capturing screenshot...")
            screenshot, num_items, products_info = self.capture_and_annotate()
            screenshot.save(f"screenshot_iter_{iteration}.png")
            print(f"✅ Screenshot saved: screenshot_iter_{iteration}.png")
            
            # Print visible products
            print(f"\n📋 Visible products:")
            for p in products_info:
                print(f"   [{p['id']}] {p.get('title', 'N/A')} - {p.get('price', 'N/A')}")
            
            # Step 2: Send to LLaVA
            print(f"\n🤖 Consulting LLaVA...")
            try:
                llava_response = self.send_to_llava(screenshot, command, num_items)
                
                if llava_response.get("success"):
                    print("\n" + "="*70)
                    print("🧠 LLAVA RESPONSE:")
                    print("="*70)
                    print(llava_response['response'])
                    print("="*70)
                    
                    # Step 3: Parse action
                    action = self.parse_action(llava_response)
                    print(f"\n📊 Parsed action: {json.dumps(action, indent=2)}")
                    
                    # Step 4: Execute action
                    success = self.execute_action(action, products_info)
                    
                    if action.get("action") == "click" and success:
                        print("\n🎉 Item clicked! Mission accomplished!")
                        action_successful = True
                        time.sleep(5)  # Wait to see the modal
                        break
                    
                    time.sleep(2)
                    
                else:
                    print(f"❌ LLaVA error: {llava_response.get('error')}")
                    break
                    
            except Exception as e:
                print(f"❌ Error: {e}")
                break
        
        print("\n" + "="*70)
        if action_successful:
            print("✅ MISSION COMPLETED!")
        else:
            print("⚠️  Max iterations reached or error occurred")
        print("="*70)
        
        input("\n👆 Press Enter to close browser...")
        self.driver.quit()

def main():
    print("=" * 70)
    print("🤖 LLaVA SHOPPING WEB AGENT")
    print("=" * 70)
    
    command = input("\n💬 Enter shopping command: ").strip()
    if not command:
        command = "buy a blue t-shirt"
        print(f"   Using default: {command}")
    
    max_iter = input("🔄 Max iterations (default 5): ").strip()
    max_iter = int(max_iter) if max_iter.isdigit() else 5
    
    agent = WebAgent()
    agent.run(command, max_iterations=max_iter)

if __name__ == "__main__":
    main()