import argparse
import threading
import time
from pathlib import Path
from PIL import Image, UnidentifiedImageError
from flask import Flask, jsonify, render_template, request
from src.evaluate import sync_gpu

def create_app(model_path, device='auto', tesseract_cmd=None, engine_factory=None):
    app = Flask(__name__)
    app.config['MAX_CONTENT_LENGTH'] = 20 * 1024 * 1024
    engines = {}
    lock = threading.Lock()

    def get_engine(method):
        if method not in engines:
            if engine_factory:
                engines[method] = engine_factory(method)
            elif method == 'donut':
                from src.donut_inference import DonutQA
                engines[method] = DonutQA(model_path, device)
            else:
                from src.ocr_qa import OCRQA
                engines[method] = OCRQA(device=device, tesseract_cmd=tesseract_cmd)
        return engines[method]

    @app.get('/')
    def home():
        return render_template('index.html')

    @app.post('/api/answer')
    def answer():
        question = request.form.get('question', '').strip()
        method = request.form.get('method', 'donut')
        upload = request.files.get('image')
        if method not in {'donut', 'ocr'} or not question or upload is None:
            return jsonify(error='Select an image and a method, and enter a question.'), 400
        if len(question) > 1000:
            return jsonify(error='The question is too long. Please shorten it.'), 400
        try:
            with Image.open(upload.stream) as img:
                image = img.convert('RGB')
            with lock:
                engine = get_engine(method)
                sync_gpu()
                start = time.perf_counter()
                result = engine.predict(image, question)
                sync_gpu()
                elapsed = time.perf_counter() - start
            return jsonify(**result, seconds=elapsed, method=method)
        except (UnidentifiedImageError, OSError) as exc:
            app.logger.exception('Image/model I/O error')
            return jsonify(error='Could not read the image or model. Check the terminal and checkpoint path.'), 400
        except ValueError as exc:
            return jsonify(error=str(exc)), 400
        except Exception:
            app.logger.exception('Inference failed')
            return jsonify(error='Model execution failed. Check the terminal for details.'), 500

    @app.errorhandler(413)
    def too_large(error):
        return jsonify(error='Image exceeds the 20 MB limit.'), 413

    return app

def main():
    parser = argparse.ArgumentParser(description='Document VQA local demo')
    parser.add_argument('--model-path', required=True)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--tesseract-cmd')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=5000)
    args = parser.parse_args()
    create_app(args.model_path, args.device, args.tesseract_cmd).run(
        host=args.host, port=args.port, debug=False, use_reloader=False)

if __name__ == '__main__':
    main()
