import subprocess
from extractor import analyze_video_file

sample_mp4 = "sample_test.mp4"

# Generate a tiny 5-second test MP4 using FFmpeg
subprocess.run([
    "ffmpeg", "-y",
    "-f", "lavfi", "-i", "testsrc=duration=5:size=720x1280:rate=30",
    "-f", "lavfi", "-i", "sine=frequency=440:duration=5",
    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac",
    sample_mp4
], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

print("Running Multimodal Feature Extractor on sample_test.mp4...")
result = analyze_video_file(sample_mp4)

print("\n--- EXTRACTED NUMERICAL FEATURE VECTOR ---")
for k, v in result["features"].items():
    print(f"  {k}: {v}")

print(f"\nKeyframes extracted: {list(result['keyframes'].keys())}")
print(f"Transcript: {result['transcript']}")