from pathlib import Path
import tempfile
from pdf2image import convert_from_path
import ollama

pdf_path = '/home/rian/python_project/myvenv/scaleup/pdf/STA-14 Wellschets.pdf'

# Original approach: use a temporary directory. The key is that ALL the work
# (saving images + sending them to Ollama) happens INSIDE the `with` block,
# before the temp directory is automatically deleted on exit.
with tempfile.TemporaryDirectory() as path:
    images = convert_from_path(pdf_path, dpi=300)

    image_paths = []
    for i, image in enumerate(images, start=1):
        img_file = Path(path) / f"page_{i}.png"
        image.save(img_file, "PNG")
        image_paths.append(str(img_file))

    print(f"Saved {len(images)} page(s) to temp dir: {path}")

    # Ollama runs here — while the temp images still exist.
    response = ollama.chat(
        model='qwen2.5vl:latest',
        messages=[
            {
                'role': 'user',
                'content': (
                    'This is an oil well completion diagram. Find every text label that '
                    'begins with the word "Sand" and the depth value written next to it. '
                    'Output ONLY a two-column markdown table with headers "Sand Name" and '
                    '"Depth". Do not add any explanation before or after the table.'
                ),
                'images': [image_paths[0]],
            }
        ],
        options={'num_ctx': 8192, 'num_predict': 2048},
    )
    print(response['message']['content'])

# Once we exit the `with` block, the temp dir and its images are gone.
print("\nTemp dir deleted — images no longer exist on disk.")
