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

# Local image generator model (downloads automatically on first run ~4GB)
GENERATOR_MODEL = "runwayml/stable-diffusion-v1-5" 

# Local vision model for reverse engineering
OLLAMA_VISION_MODEL = "llava" 

# --- GITHUB CONFIGURATION ---
GITHUB_REPO_PATH = "." 
GITHUB_IMAGES_TARGET_DIR = "images/spider" # Target folder inside your GitHub repository where images are stored
GITHUB_BRANCH = "main"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def initialize_generator():
    print("⏳ Loading local AI Image Generator into memory (this will take a moment on first boot)...")
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    
    pipe = StableDiffusionPipeline.from_pretrained(
        GENERATOR_MODEL, 
        torch_dtype=dtype
    )
    pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")
    return pipe

def batch_generate_images(pipe, base_prompt, num_images=2):
    """Generates multiple images locally from a single master prompt with isolated random generators."""
    print(f"🎨 Generating {num_images} images for prompt: '{base_prompt}'...")
    image_paths = []
    
    for i in range(num_images):
        # Create an explicit random generator for each batch item to ensure clean noise initialization
        device_type = "cuda" if torch.cuda.is_available() else "cpu"
        gen = torch.Generator(device=device_type).manual_seed(torch.seed())
        
        result = pipe(base_prompt, generator=gen)
        image = result.images[0]
        
        # Save image locally
        file_name = f"gen_{i+1}.png"
        file_path = os.path.join(OUTPUT_DIR, file_name)
        image.save(file_path)
        image_paths.append(file_path)
        print(f" -> Saved image: {file_path}")
        
    return image_paths

def reverse_engineer_prompts(image_paths):
    """Passes each generated image into Ollama to write a clean prompt back."""
    print("🔍 Analyzing images with local Ollama vision model...")
    dataset_entries = []
    
    analysis_instruction = (
        "Analyze this image closely. Write a detailed, descriptive text-to-image prompt "
        "that could be fed back into an AI image generator to recreate this exact picture. "
        "Focus on subject matter, composition, lighting, colors, style, and background details. "
        "Return ONLY the raw prompt text, without any conversational filler."
    )

    for path in image_paths:
        try:
            response = ollama.chat(
                model=OLLAMA_VISION_MODEL,
                messages=[
                    {
                        'role': 'user',
                        'content': analysis_instruction,
                        'images': [path]
                    }
                ]
            )
            
            reversed_prompt = response['message']['content'].strip()
            
            # Formats the hosted URL schema pointing to your target repo structure
            file_name = os.path.basename(path)
            hosted_url = f"https://raw.githubusercontent.com/DasStudioINC/ALLI_AI_IMAGE_TRAIN_REPO/main/{GITHUB_IMAGES_TARGET_DIR}/{file_name}"
            
            dataset_entries.append({
                "url": hosted_url,
                "prompt": reversed_prompt
            })
            print(f" ✔ Reverse-engineered prompt for: {path}")
            
        except Exception as e:
            print(f" ❌ Error processing {path}: {e}")
            
    return dataset_entries

def save_to_json(data, json_path):
    """Saves structured data into your requested JSON format schema."""
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)
    print(f"💾 Successfully saved dataset schema to {json_path}")

def push_to_github(local_entries, status_callback=lambda msg: None):
    """Pulls remote repo changes, appends local entries to dataset.json, copies images, and commits/pushes."""
    status_callback("Connecting to Git repository...")
    try:
        repo = Repo(GITHUB_REPO_PATH)
        
        # 1. Pull latest changes from GitHub to prevent merge conflicts
        status_callback("Pulling latest changes from GitHub...")
        origin = repo.remotes.origin
        origin.pull(GITHUB_BRANCH)
        
        # 2. Load existing repository dataset.json if it exists and append new entries
        repo_json_path = os.path.join(GITHUB_REPO_PATH, JSON_OUTPUT_PATH)
        existing_data = []
        if os.path.exists(repo_json_path):
            with open(repo_json_path, "r", encoding="utf-8") as f:
                try:
                    existing_data = json.load(f)
                except json.JSONDecodeError:
                    existing_data = []
                    
        combined_data = existing_data + local_entries
        
        with open(repo_json_path, "w", encoding="utf-8") as f:
            json.dump(combined_data, f, indent=4)
            
        # 3. Copy generated images from local output folder to the repository target image directory
        target_img_folder = os.path.join(GITHUB_REPO_PATH, GITHUB_IMAGES_TARGET_DIR)
        os.makedirs(target_img_folder, exist_ok=True)
        
        for entry in local_entries:
            filename = entry["url"].split("/")[-1]
            src_path = os.path.join(OUTPUT_DIR, filename)
            dst_path = os.path.join(target_img_folder, filename)
            if os.path.exists(src_path):
                shutil.copy(src_path, dst_path)
                
        # 4. Stage, commit, and push updates back to GitHub
        status_callback("Staging and pushing changes to GitHub...")
        repo.index.add([JSON_OUTPUT_PATH, GITHUB_IMAGES_TARGET_DIR])
        repo.index.commit("Auto-update dataset and generated images via Bach AI app")
        origin.push(GITHUB_BRANCH)
        
        status_callback("Successfully pushed dataset and images to GitHub!")
        return True
    except Exception as e:
        error_msg = str(e)
        status_callback(f"Git Push Error: {error_msg}")
        print(f"❌ Git Push Error: {error_msg}")
        return False