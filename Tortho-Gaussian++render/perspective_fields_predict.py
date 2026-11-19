import argparse
import json
import math
import os
import sys


def parse_args():
    parser = argparse.ArgumentParser(
        description="Predict roll and pitch with PerspectiveFields in the current Python environment."
    )
    parser.add_argument("--image", default=None, type=str)
    parser.add_argument("--input-json", default=None, type=str)
    parser.add_argument("--output", required=True, type=str)
    parser.add_argument("--perspective-fields-path", required=True, type=str)
    parser.add_argument("--version", required=True, type=str)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--batch-size", default=1, type=int)
    return parser.parse_args()


def load_input_items(args):
    if bool(args.image) == bool(args.input_json):
        raise ValueError("Pass exactly one of --image or --input-json.")

    if args.image:
        image_path = os.path.abspath(os.path.expanduser(args.image))
        return [{"key": "0", "path": image_path}], False

    input_json_path = os.path.abspath(os.path.expanduser(args.input_json))
    if not os.path.isfile(input_json_path):
        raise FileNotFoundError(f"Batch input JSON does not exist: {input_json_path}")

    with open(input_json_path, "r", encoding="utf-8") as input_file:
        payload = json.load(input_file)
    items = payload.get("images")
    if not isinstance(items, list) or len(items) == 0:
        raise ValueError("Batch input JSON must contain a non-empty 'images' list.")

    normalized_items = []
    seen_keys = set()
    for index, item in enumerate(items):
        key = str(item.get("key", index))
        if key in seen_keys:
            raise ValueError(f"Duplicate image key in batch input: {key}")
        seen_keys.add(key)
        image_path = os.path.abspath(os.path.expanduser(str(item["path"])))
        normalized_items.append({"key": key, "path": image_path})
    return normalized_items, True


def predict_batch(model, image_items, batch_size, cv2):
    predictions = []
    total = len(image_items)
    for batch_start in range(0, total, batch_size):
        current_items = image_items[batch_start : batch_start + batch_size]
        images_bgr = []
        for item in current_items:
            image_path = item["path"]
            if not os.path.isfile(image_path):
                raise FileNotFoundError(f"Input image does not exist: {image_path}")
            image_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
            if image_bgr is None:
                raise ValueError(f"OpenCV could not decode input image: {image_path}")
            images_bgr.append(image_bgr)

        if len(images_bgr) == 1:
            batch_predictions = [model.inference(img_bgr=images_bgr[0])]
        else:
            batch_predictions = model.inference_batch(img_bgr_list=images_bgr)

        for item, prediction in zip(current_items, batch_predictions):
            roll_degrees = float(prediction["pred_roll"].item())
            pitch_degrees = float(prediction["pred_pitch"].item())
            if not math.isfinite(roll_degrees) or not math.isfinite(pitch_degrees):
                raise ValueError(
                    "PerspectiveFields returned non-finite roll/pitch values: "
                    f"roll={roll_degrees}, pitch={pitch_degrees}, "
                    f"image={item['path']}"
                )
            predictions.append(
                {
                    "key": item["key"],
                    "path": item["path"],
                    "roll": roll_degrees,
                    "pitch": pitch_degrees,
                }
            )
            print(
                f"[{len(predictions)}/{total}] {item['key']}: "
                f"roll={roll_degrees:.6f} deg, pitch={pitch_degrees:.6f} deg",
                flush=True,
            )
    return predictions


def main():
    args = parse_args()
    output_path = os.path.abspath(os.path.expanduser(args.output))
    perspective_fields_path = os.path.abspath(
        os.path.expanduser(args.perspective_fields_path)
    )
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1.")

    if not os.path.isdir(perspective_fields_path):
        raise FileNotFoundError(
            f"PerspectiveFields project directory does not exist: "
            f"{perspective_fields_path}"
        )
    image_items, is_batch = load_input_items(args)

    sys.path.insert(0, perspective_fields_path)

    import cv2
    import torch
    from perspective2d import PerspectiveFields

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "PerspectiveFields subprocess requested CUDA, but CUDA is unavailable."
        )

    print(
        f"Loading model '{args.version}' on {args.device} "
        f"from {perspective_fields_path}",
        flush=True,
    )
    model = PerspectiveFields(args.version).eval().to(args.device)
    if not model.param_on:
        raise ValueError(
            f"PerspectiveFields model '{args.version}' does not predict roll/pitch."
        )

    predictions = predict_batch(model, image_items, args.batch_size, cv2)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as output_file:
        if is_batch:
            json.dump({"predictions": predictions}, output_file)
        else:
            prediction = predictions[0]
            json.dump(
                {
                    "roll": prediction["roll"],
                    "pitch": prediction["pitch"],
                },
                output_file,
            )

    print(f"Prediction complete: {len(predictions)} image(s)", flush=True)


if __name__ == "__main__":
    main()
