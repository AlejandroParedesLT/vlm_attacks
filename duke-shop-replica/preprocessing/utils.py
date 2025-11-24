import os
import requests
from bs4 import BeautifulSoup

# Target URL
base_url = "https://shop.duke.edu/gifts/automotive/keychains-and-lanyards"

# Folder to save images
save_folder = "duke_mens_tees"
os.makedirs(save_folder, exist_ok=True)

# Fetch the page
response = requests.get(base_url)
soup = BeautifulSoup(response.text, "html.parser")

# Inspect and find all product images
# Usually, images are in <img> tags with src or data-src attributes
image_tags = soup.find_all("img")

for i, img in enumerate(image_tags):
    # Get the image URL
    img_url = img.get("src") or img.get("data-src")
    if not img_url:
        continue
    if img_url.startswith("//"):
        img_url = "https:" + img_url
    elif img_url.startswith("/"):
        img_url = "https://shop.duke.edu" + img_url

    # Download the image
    try:
        img_data = requests.get(img_url).content
        with open(os.path.join(save_folder, f"image_{i}.jpg"), "wb") as f:
            f.write(img_data)
        print(f"Downloaded {img_url}")
    except Exception as e:
        print(f"Failed {img_url}: {e}")
