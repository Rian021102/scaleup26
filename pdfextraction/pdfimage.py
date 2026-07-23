from pathlib import Path
from pdf2image import convert_from_path
import ollama

BASE = Path(__file__).resolve().parent.parent
pdf_path = BASE / "pdf" / "STA-14 Wellschets.pdf"
out_dir = BASE / "pdf" / "images"
out_dir.mkdir(parents=True, exist_ok=True)

images = convert_from_path(pdf_path, dpi=300)

for i, image in enumerate(images, start=1):
    image.save(out_dir / f"page_{i}.png", "PNG")

print(f"Saved {len(images)} page(s) to {out_dir}")




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
            'images': [str(out_dir / "page_1.png")]
        }
    ],
    options={'num_ctx': 8192, 'num_predict': 2048}
)
print(response['message']['content'])