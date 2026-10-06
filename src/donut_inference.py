from pathlib import Path

class DonutQA:
    def __init__(self, model_path, device='auto'):
        import torch
        from transformers import DonutProcessor, VisionEncoderDecoderModel
        self.torch = torch
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device == 'auto' else torch.device(device)
        self.processor = DonutProcessor.from_pretrained(model_path, use_fast=False)
        self.model = VisionEncoderDecoderModel.from_pretrained(model_path).to(self.device)
        self.model.eval()

    def predict(self, image, question):
        p = self.processor
        prompt = f'<s_docvqa><s_question>{question}</s_question><s_answer>'
        ids = p.tokenizer(prompt, add_special_tokens=False, return_tensors='pt').input_ids.to(self.device)
        if ids.shape[1] >= 128:
            raise ValueError('Question exceeds the decoder token limit. Please shorten it.')
        pixels = p(images=image.convert('RGB'), return_tensors='pt').pixel_values.to(self.device)
        with self.torch.inference_mode():
            output = self.model.generate(
                pixel_values=pixels, decoder_input_ids=ids, max_length=128,
                do_sample=False, num_beams=1, pad_token_id=p.tokenizer.pad_token_id,
                eos_token_id=p.tokenizer.eos_token_id,
                bad_words_ids=[[p.tokenizer.unk_token_id]], use_cache=True)
        return {'answer': p.tokenizer.decode(output[0, ids.shape[1]:], skip_special_tokens=True).strip()}
