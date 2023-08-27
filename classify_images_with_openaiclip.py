"""
Image Classification with CLIP (Contrastive Language-Image Pre-training)

This script provides utilities to classify images in a given folder against
a set of predefined labels using the CLIP model. The classified images are
then organized into separate folders based on their labels.

Features:
- Uses CLIP model for image classification.
- Allows concurrent processing for faster classification of multiple images.
- Option to record inference results in a CSV file.
- Command Line Interface (CLI) support for easy execution.

Usage:
    python script_name.py --target-folder path_to_images_folder
                          --output-folder path_to_save_classified_images
                          [--parallel num_of_threads]
                          [--inference]
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
from tqdm import tqdm
import os
import shutil
import torch
import click
import csv


def classify_single_image(filename, folder_path, labels, model, processor):
    """Classify a single image against the given set of labels."""
    image_path = os.path.join(folder_path, filename)
    image = Image.open(image_path)

    text_inputs = processor(
        text=labels, images=image, return_tensors="pt",
        padding=True, truncation=True
    )

    outputs = model(**text_inputs)
    image_features = outputs.image_embeds
    text_features = outputs.text_embeds

    image_features = torch.nn.functional.normalize(image_features, p=2, dim=-1)
    text_features = torch.nn.functional.normalize(text_features, p=2, dim=-1)

    similarity = (image_features @ text_features.T).softmax(dim=-1)
    values, indices = torch.topk(similarity, k=1, dim=-1)

    chosen_label = labels[indices.item()]
    confidence = values.item()

    return chosen_label, filename, confidence


def classify_images(folder_path, labels, output_folder, num_threads=1,
                    record_inference=False):
    """Classify multiple images in a folder against a set of labels."""
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")

    classified_images = {label: [] for label in labels}

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [
            executor.submit(
                classify_single_image, filename, folder_path, labels,
                model, processor
            )
            for filename in os.listdir(folder_path)
            if filename.endswith(('.png', '.jpg', '.jpeg'))
        ]

        if record_inference:
            with open(os.path.join(output_folder, "inference_results.csv"),
                      "w", newline='') as csvfile:
                csvwriter = csv.writer(csvfile)
                csvwriter.writerow(["Filename", "Label", "Confidence"])

        for future in tqdm(as_completed(futures), total=len(futures),
                   desc="Processing images"):
            chosen_label, filename, confidence = future.result()
            classified_images[chosen_label].append(filename)

    if record_inference:
        csv_file_path = os.path.join(output_folder, "inference_results.csv")
        with open(csv_file_path, "a", newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow([filename, chosen_label, confidence])

            if record_inference:
                with open(os.path.join(output_folder, "inference_results.csv"),
                          "a", newline='') as csvfile:
                    csvwriter = csv.writer(csvfile)
                    csvwriter.writerow([filename, chosen_label, confidence])

        for label, images in classified_images.items():
            label_folder = os.path.join(output_folder, label)
            if not os.path.exists(label_folder):
                os.makedirs(label_folder)

            for image in images:
                src = os.path.join(folder_path, image)
                dest = os.path.join(label_folder, image)
                shutil.copy(src, dest)


@click.command()
@click.option('--target-folder', required=True, type=click.Path(exists=True),
              help="Path to folder containing images to process.")
@click.option('--output-folder', required=True, type=click.Path(),
              help="Path to folder where the classified images will be saved.")
@click.option('--parallel', default=8, type=int,
              help='Number of parallel threads to use.')
@click.option('--inference', is_flag=True,
              help='Record inference results to CSV.')
def cli(target_folder, output_folder, parallel, inference):
    """Entry point for the CLI application."""
    labels = ["dragon", "ghoul", "cyclop", "humbaba", "namazu",
              "a-senee-ki-wakw", "longma", "bashee", "imp"]
    classify_images(
        target_folder, labels, output_folder, num_threads=parallel,
        record_inference=inference
    )


if __name__ == '__main__':
    cli()
