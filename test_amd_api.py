import os
import base64
from io import BytesIO
from PIL import Image, ImageDraw
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    api_key=os.getenv("AMD_API_KEY"),
    base_url=os.getenv("AMD_BASE_URL", "https://developer.amd.com.cn/radeon/api/v1")
)

print("1. Fetching exact model IDs from AMD server...")
try:
    models = client.models.list()
    available_ids = [m.id for m in models.data]
    print("Available Model IDs:", available_ids)
except Exception as e:
    print("Could not list models directly (normal on some gateways):", e)

print("\n2. Testing Text Reasoning (DeepSeek-V4.1-Flash)...")
try:
    resp = client.chat.completions.create(
        model="DeepSeek-V4.1-Flash",
        messages=[{"role": "user", "content": "Give me a 1-sentence viral hook for a tech video about AI agents."}],
        max_tokens=100
    )
    print("Text Response:", resp.choices[0].message.content)
except Exception as e:
    print("Text test error:", e)

print("\n3. Testing Multimodal Vision (MiniCPM-V-4.7-35B-A3B)...")
try:
    # Create a tiny synthetic test frame with low-contrast text
    img = Image.new("RGB", (400, 200), color=(30, 30, 30))
    draw = ImageDraw.Draw(img)
    draw.text((20, 80), "SUBSCRIBE NOW", fill=(70, 70, 70)) # Low contrast text
    buf = BytesIO()
    img.save(buf, format="JPEG")
    b64_img = base64.b64encode(buf.getvalue()).decode("utf-8")

    vis_resp = client.chat.completions.create(
        model="MiniCPM-V-4.7-35B-A3B",
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": "Read the text in this video frame and critique its contrast/legibility in 1 sentence."},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"}}
            ]
        }],
        max_tokens=120
    )
    print("Vision Response:", vis_resp.choices[0].message.content)
except Exception as e:
    print("Vision test error:", e)
