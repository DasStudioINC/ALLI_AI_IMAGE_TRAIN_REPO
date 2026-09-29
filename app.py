import os
import json
import shutil
from pathlib import Path
from PIL import Image
import torch
from diffusers import StableDiffusionPipeline
import ollama
from git import Repo

# --- CONFIGURATION ---
OUTPUT_DIR = "generated_dataset"
JSON_OUTPUT_PATH = "dataset.json"

# Local image generator model
GENERATOR_MODEL = "runwayml/stable-diffusion-v1-5"

# Local vision model for reverse engineering
OLLAMA_VISION_MODEL = "llava"

# --- GITHUB CONFIGURATION ---
GITHUB_REPO_PATH = "."
GITHUB_IMAGES_TARGET_DIR = "images"
GITHUB_BRANCH = "main"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def initialize_generator():
    print(
        "⏳ Loading local AI Image Generator into memory "
        "(this will take a moment on first boot)..."
    )

    dtype = (
        torch.float16
        if torch.cuda.is_available()
        else torch.float32
    )

    pipe = StableDiffusionPipeline.from_pretrained(
        GENERATOR_MODEL,
        torch_dtype=dtype
    )

    pipe = pipe.to(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    return pipe


def batch_generate_images(pipe, base_prompt, num_images=2):
    """
    Generates multiple images locally from a single master prompt
    with isolated random generators.
    """

    print(
        f"🎨 Generating {num_images} images for prompt: "
        f"'{base_prompt}'..."
    )

    image_paths = []

    # ---------------------------------------------------------
    # Load generation counter
    # ---------------------------------------------------------

    genCount = 0

    if os.path.exists("gen.txt"):
        with open("gen.txt", "r") as file:
            content = file.read().strip()

            if content:
                try:
                    genCount = int(content.split(":")[1])
                except (ValueError, IndexError):
                    print(
                        "⚠️ Invalid gen.txt format. "
                        "Starting generation count at 0."
                    )
                    genCount = 0

    # ---------------------------------------------------------
    # Generate images
    # ---------------------------------------------------------

    for i in range(num_images):

        device_type = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        gen = torch.Generator(
            device=device_type
        ).manual_seed(
            torch.seed()
        )

        result = pipe(
            base_prompt,
            generator=gen
        )

        image = result.images[0]

        # Increment generation number
        genCount += 1

        file_name = f"gen_{genCount}.png"

        file_path = os.path.join(
            OUTPUT_DIR,
            file_name
        )

        image.save(file_path)

        image_paths.append(file_path)

        print(
            f" -> Saved image: {file_path}"
        )

    # ---------------------------------------------------------
    # Save updated generation counter
    # ---------------------------------------------------------

    with open("gen.txt", "w") as file:
        file.write(f"gen:{genCount}")

    return image_paths


def reverse_engineer_prompts(image_paths):
    """
    Passes each generated image into Ollama to write
    a clean prompt back.
    """

    print(
        "🔍 Analyzing images with local Ollama vision model..."
    )

    dataset_entries = []

    analysis_instruction = (
        "Analyze this image closely. Write a detailed, "
        "descriptive text-to-image prompt that could be fed "
        "back into an AI image generator to recreate this "
        "exact picture. Focus on subject matter, composition, "
        "lighting, colors, style, and background details. "
        "Return ONLY the raw prompt text, without any "
        "conversational filler."
    )

    for path in image_paths:

        try:

            response = ollama.chat(
                model=OLLAMA_VISION_MODEL,
                messages=[
                    {
                        "role": "user",
                        "content": analysis_instruction,
                        "images": [path]
                    }
                ]
            )

            reversed_prompt = (
                response["message"]["content"].strip()
            )

            # -------------------------------------------------
            # Build GitHub hosted URL
            # -------------------------------------------------

            file_name = os.path.basename(path)

            hosted_url = (
                "https://raw.githubusercontent.com/"
                "DasStudioINC/"
                "ALLI_AI_IMAGE_TRAIN_REPO/"
                f"{GITHUB_BRANCH}/"
                f"{GITHUB_IMAGES_TARGET_DIR}/"
                f"{file_name}"
            )

            dataset_entries.append({
                "url": hosted_url,
                "prompt": reversed_prompt
            })

            print(
                f" ✔ Reverse-engineered prompt for: {path}"
            )

        except Exception as e:

            print(
                f" ❌ Error processing {path}: {e}"
            )

    return dataset_entries


def save_to_json(data, json_path):
    """
    Append new entries to dataset.json without deleting
    existing entries.

    Existing entries are preserved.

    New entries are added only if their URL does not
    already exist in the dataset.
    """

    # ---------------------------------------------------------
    # Load existing dataset
    # ---------------------------------------------------------

    existing_data = []

    if os.path.exists(json_path):

        try:

            with open(
                json_path,
                "r",
                encoding="utf-8"
            ) as f:

                existing_data = json.load(f)

                if not isinstance(existing_data, list):
                    print(
                        "⚠️ Existing dataset.json is not a list. "
                        "Starting with an empty dataset."
                    )

                    existing_data = []

        except json.JSONDecodeError:

            print(
                "⚠️ Existing dataset.json contains invalid JSON. "
                "Starting with an empty dataset."
            )

            existing_data = []

    # ---------------------------------------------------------
    # Find existing URLs
    # ---------------------------------------------------------

    existing_urls = {
        entry.get("url")
        for entry in existing_data
        if isinstance(entry, dict)
    }

    # ---------------------------------------------------------
    # Only add new entries
    # ---------------------------------------------------------

    new_unique_entries = []

    for entry in data:

        if not isinstance(entry, dict):
            continue

        url = entry.get("url")

        if url not in existing_urls:

            new_unique_entries.append(entry)

            existing_urls.add(url)

    # ---------------------------------------------------------
    # Combine old + new
    # ---------------------------------------------------------

    combined_data = (
        existing_data +
        new_unique_entries
    )

    # ---------------------------------------------------------
    # Write combined dataset
    # ---------------------------------------------------------

    with open(
        json_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            combined_data,
            f,
            indent=4
        )

    print(
        f"💾 Successfully updated {json_path}"
    )

    print(
        f"   Existing entries: {len(existing_data)}"
    )

    print(
        f"   New entries added: {len(new_unique_entries)}"
    )

    print(
        f"   Total entries: {len(combined_data)}"
    )

    return new_unique_entries


def push_to_github(
    local_entries,
    status_callback=lambda msg: None
):
    """
    Pulls the latest GitHub repository changes,
    appends unique local entries to dataset.json,
    copies generated images, and commits/pushes.
    """

    status_callback(
        "Connecting to Git repository..."
    )

    try:

        repo = Repo(GITHUB_REPO_PATH)

        # =====================================================
        # 1. Pull latest changes
        # =====================================================

        status_callback(
            "Pulling latest changes from GitHub..."
        )

        origin = repo.remotes.origin

        origin.pull(GITHUB_BRANCH)

        # =====================================================
        # 2. Load existing dataset.json
        # =====================================================

        repo_json_path = os.path.join(
            GITHUB_REPO_PATH,
            JSON_OUTPUT_PATH
        )

        existing_data = []

        if os.path.exists(repo_json_path):

            with open(
                repo_json_path,
                "r",
                encoding="utf-8"
            ) as f:

                try:

                    existing_data = json.load(f)

                    if not isinstance(
                        existing_data,
                        list
                    ):
                        existing_data = []

                except json.JSONDecodeError:

                    print(
                        "⚠️ GitHub dataset.json contains "
                        "invalid JSON."
                    )

                    existing_data = []

        # =====================================================
        # 3. Find existing URLs
        # =====================================================

        existing_urls = {
            entry.get("url")
            for entry in existing_data
            if isinstance(entry, dict)
        }

        # =====================================================
        # 4. Add only unique entries
        # =====================================================

        new_unique_entries = []

        for entry in local_entries:

            if not isinstance(entry, dict):
                continue

            url = entry.get("url")

            if url not in existing_urls:

                new_unique_entries.append(entry)

                existing_urls.add(url)

        combined_data = (
            existing_data +
            new_unique_entries
        )

        # =====================================================
        # 5. Write dataset.json
        # =====================================================

        with open(
            repo_json_path,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                combined_data,
                f,
                indent=4
            )

        print(
            f"📄 Dataset updated:"
        )

        print(
            f"   Existing entries: "
            f"{len(existing_data)}"
        )

        print(
            f"   New entries: "
            f"{len(new_unique_entries)}"
        )

        print(
            f"   Total entries: "
            f"{len(combined_data)}"
        )

        # =====================================================
        # 6. Copy generated images
        # =====================================================

        target_img_folder = os.path.join(
            GITHUB_REPO_PATH,
            GITHUB_IMAGES_TARGET_DIR
        )

        os.makedirs(
            target_img_folder,
            exist_ok=True
        )

        for entry in local_entries:

            filename = entry["url"].split("/")[-1]

            src_path = os.path.join(
                OUTPUT_DIR,
                filename
            )

            dst_path = os.path.join(
                target_img_folder,
                filename
            )

            if os.path.exists(src_path):

                shutil.copy(
                    src_path,
                    dst_path
                )

                print(
                    f"📷 Copied image: {filename}"
                )

        # =====================================================
        # 7. Check whether there is anything to commit
        # =====================================================

        if not new_unique_entries and not local_entries:

            status_callback(
                "Nothing new to push."
            )

            return True

        # =====================================================
        # 8. Stage changes
        # =====================================================

        status_callback(
            "Staging and pushing changes to GitHub..."
        )

        repo.index.add([
            JSON_OUTPUT_PATH,
            GITHUB_IMAGES_TARGET_DIR
        ])

        # =====================================================
        # 9. Commit
        # =====================================================

        # Check whether there are actually changes
        if repo.is_dirty(
            untracked_files=True
        ):

            repo.index.commit(
                "Auto-update dataset and generated images "
                "via Bach AI app"
            )

            # =================================================
            # 10. Push
            # =================================================

            origin.push(
                GITHUB_BRANCH
            )

            status_callback(
                "Successfully pushed dataset and images "
                "to GitHub!"
            )

        else:

            status_callback(
                "No changes detected. Nothing to push."
            )

        return True

    except Exception as e:

        error_msg = str(e)

        status_callback(
            f"Git Push Error: {error_msg}"
        )

        print(
            f"❌ Git Push Error: {error_msg}"
        )

        return False