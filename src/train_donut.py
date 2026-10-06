import argparse
from pathlib import Path
from src.common import read_json, save_json

def main():
    parser = argparse.ArgumentParser(description='Decoder-only Donut fine-tuning (CUDA required)')
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--model-path', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--results-dir', required=True)
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--learning-rate', type=float, default=1e-5)
    args = parser.parse_args()
    import torch
    from transformers import DonutProcessor, VisionEncoderDecoderModel
    if not torch.cuda.is_available():
        raise RuntimeError('Training requires a CUDA GPU. Use Colab GPU.')
    DATA_DIR = Path(args.data_dir)
    RESULT_DIR = Path(args.results_dir)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    train_rows = read_json(DATA_DIR / 'train.json')
    processor = DonutProcessor.from_pretrained(args.model_path, use_fast=False)
    model = VisionEncoderDecoderModel.from_pretrained(args.model_path).to('cuda')
    import json
    import random
    import math
    import torch
    from PIL import Image
    from tqdm.auto import tqdm

    SEED = 42
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    with open(DATA_DIR / "validation.json", encoding="utf-8") as f:
        validation_rows = json.load(f)

    MAX_LENGTH = 128
    EPOCHS = args.epochs
    ACCUMULATION = 4
    LEARNING_RATE = args.learning_rate

    CHECKPOINT_DIR = Path(args.output_dir)
    if any(CHECKPOINT_DIR.glob('*.safetensors')) or (CHECKPOINT_DIR / 'pytorch_model.bin').exists():
        raise FileExistsError('Output already contains weights. Choose a new output directory.')
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    def encode_text(row):
        prompt = (
            f"<s_docvqa><s_question>{row['question']}"
            "</s_question><s_answer>"
        )
        prompt_ids = processor.tokenizer(
            prompt, add_special_tokens=False
        ).input_ids

        target_ids = processor.tokenizer(
            row["answers"][0] + "</s_answer>",
            add_special_tokens=False
        ).input_ids

        ids = prompt_ids + target_ids + [processor.tokenizer.eos_token_id]
        return prompt_ids, ids

    def valid_length(row):
        _, ids = encode_text(row)
        return len(ids) <= MAX_LENGTH

    # Skip overly long samples instead of truncating answers
    training_data = [r for r in train_rows if valid_length(r)]
    validation_data = [r for r in validation_rows if valid_length(r)]

    print("Usable training samples:", len(training_data), "/", len(train_rows))
    print("Usable validation samples:", len(validation_data), "/", len(validation_rows))
    save_json(RESULT_DIR / 'training_config.json', {
        'seed': SEED, 'epochs': EPOCHS, 'learning_rate': LEARNING_RATE,
        'batch_size': 1, 'accumulation': ACCUMULATION, 'max_length': MAX_LENGTH,
        'weight_decay': 0.01, 'encoder_frozen': True, 'mixed_precision': 'fp16',
        'train_used': len(training_data), 'validation_used': len(validation_data),
        'initial_model': args.model_path,
    })

    assert training_data and validation_data

    def make_inputs(row):
        prompt_ids, ids = encode_text(row)

        decoder_ids = torch.tensor([ids[:-1]], device="cuda")
        labels = torch.tensor([ids[1:]], device="cuda")
        labels[:, :len(prompt_ids) - 1] = -100

        with Image.open(DATA_DIR / row["image_path"]) as img:
            pixels = processor(
                images=img.convert("RGB"),
                return_tensors="pt"
            ).pixel_values.to("cuda")

        return pixels, decoder_ids, labels

    for p in model.encoder.parameters():
        p.requires_grad = False
    for p in model.decoder.parameters():
        p.requires_grad = True

    trainable_parameters = [p for p in model.parameters() if p.requires_grad]

    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=LEARNING_RATE,
        weight_decay=0.01
    )
    scaler = torch.amp.GradScaler("cuda")

    def validation_loss():
        model.eval()
        total_loss = 0.0
        total_tokens = 0

        with torch.inference_mode():
            for row in tqdm(validation_data, desc="Validation"):
                pixels, decoder_ids, labels = make_inputs(row)
                token_count = (labels != -100).sum().item()

                with torch.autocast("cuda", dtype=torch.float16):
                    output = model(
                        pixel_values=pixels,
                        decoder_input_ids=decoder_ids,
                        labels=labels,
                        use_cache=False
                    )

                total_loss += output.loss.item() * token_count
                total_tokens += token_count

        return total_loss / total_tokens

    # Save the initial baseline so a worse fine-tuned model need not be selected
    best_validation_loss = validation_loss()
    model.save_pretrained(CHECKPOINT_DIR, safe_serialization=True)
    processor.save_pretrained(CHECKPOINT_DIR)

    history = [{
        "epoch": 0,
        "validation_loss": best_validation_loss
    }]
    print("Validation loss before fine-tuning:", best_validation_loss)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        model.encoder.eval()

        order = training_data.copy()
        random.shuffle(order)
        optimizer.zero_grad(set_to_none=True)

        total_loss = 0.0
        total_tokens = 0
        progress = tqdm(range(len(order)), desc=f"Epoch {epoch}/{EPOCHS}")

        for index in progress:
            row = order[index]
            pixels, decoder_ids, labels = make_inputs(row)
            token_count = (labels != -100).sum().item()

            # The last group may contain fewer than 4 samples
            group_start = (index // ACCUMULATION) * ACCUMULATION
            group_size = min(ACCUMULATION, len(order) - group_start)

            with torch.autocast("cuda", dtype=torch.float16):
                output = model(
                    pixel_values=pixels,
                    decoder_input_ids=decoder_ids,
                    labels=labels,
                    use_cache=False
                )
                loss = output.loss

            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss for sample {row['question_id']}")

            total_loss += loss.item() * token_count
            total_tokens += token_count
            scaler.scale(loss / group_size).backward()

            if (index + 1) % ACCUMULATION == 0 or index + 1 == len(order):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable_parameters, 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            progress.set_postfix(loss=f"{loss.item():.3f}")
            del output, loss, pixels, decoder_ids, labels

        train_loss = total_loss / total_tokens
        val_loss = validation_loss()

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_loss": val_loss
        })

        print(f"Epoch {epoch}: train={train_loss:.4f}, validation={val_loss:.4f}")

        if val_loss < best_validation_loss:
            best_validation_loss = val_loss
            model.save_pretrained(CHECKPOINT_DIR, safe_serialization=True)
            processor.save_pretrained(CHECKPOINT_DIR)
            print("Saved an improved checkpoint.")

        (RESULT_DIR / "training_history.json").write_text(
            json.dumps(history, indent=2),
            encoding="utf-8"
        )

    model.eval()
    print("Completed. Selected checkpoint:", CHECKPOINT_DIR)
    print("History:", history)
if __name__ == '__main__':
    main()
