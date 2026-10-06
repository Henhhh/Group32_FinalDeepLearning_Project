import io
import json
import os
from pathlib import Path
from PIL import Image

def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    os.replace(temporary, path)

def to_image(value):
    if isinstance(value, Image.Image):
        return value.convert('RGB')
    if isinstance(value, bytes):
        return Image.open(io.BytesIO(value)).convert('RGB')
    if value.get('bytes') is not None:
        return Image.open(io.BytesIO(value['bytes'])).convert('RGB')
    return Image.open(value['path']).convert('RGB')

def degrade_image(image):
    image = image.convert('RGB')
    w, h = image.size
    small = image.resize((max(1,w//2), max(1,h//2)), Image.Resampling.LANCZOS)
    return small.resize((w,h), Image.Resampling.BICUBIC)
