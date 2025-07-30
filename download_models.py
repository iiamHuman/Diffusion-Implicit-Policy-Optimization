#!/usr/bin/env python3
"""
HuggingFace Model Downloader Script
Downloads specified models to local directories
"""

import os
from pathlib import Path
from huggingface_hub import snapshot_download
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define models to download
MODELS = [
    {
        "repo_id": "laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
        "local_dir": "./models/CLIP-ViT-H-14-laion2B-s32B-b79K"
    },
    {
        "repo_id": "openai/clip-vit-base-patch32", 
        "local_dir": "./models/clip-vit-base-patch32"
    },
    {
        "repo_id": "xswu/HPSv2-hf",
        "local_dir": "./models/HPSv2-hf"
    }
]

def download_model(repo_id, local_dir, token=None):
    """
    Download a single model
    
    Args:
        repo_id (str): HuggingFace model repository ID
        local_dir (str): Local directory to save the model
        token (str, optional): HuggingFace access token
    """
    try:
        logger.info(f"Starting download for model: {repo_id}")
        logger.info(f"Saving to: {local_dir}")
        
        # Create local directory
        Path(local_dir).mkdir(parents=True, exist_ok=True)
        
        # Download model
        snapshot_download(
            repo_id=repo_id,
            local_dir=local_dir,
            token=token,
            resume_download=True,  # Support resume download
            local_dir_use_symlinks=False  # Don't use symlinks, copy files directly
        )
        
        logger.info(f"✅ Model download completed: {repo_id}")
        
    except Exception as e:
        logger.error(f"❌ Failed to download model {repo_id}: {str(e)}")
        raise

def main():
    """Main function"""
    # Optional: Set HuggingFace access token (if needed for private models)
    # token = "your_huggingface_token_here"
    token = None
    
    # Use token from environment variable if available
    if "HUGGINGFACE_TOKEN" in os.environ:
        token = os.environ["HUGGINGFACE_TOKEN"]
        logger.info("Using HuggingFace token from environment variable")
    
    logger.info("Starting batch model download...")
    logger.info(f"Planning to download {len(MODELS)} models")
    
    success_count = 0
    failed_models = []
    
    for i, model in enumerate(MODELS, 1):
        try:
            logger.info(f"\n[{i}/{len(MODELS)}] Processing model...")
            download_model(
                repo_id=model["repo_id"],
                local_dir=model["local_dir"],
                token=token
            )
            success_count += 1
            
        except Exception as e:
            failed_models.append(model["repo_id"])
            logger.error(f"Skipping failed model: {model['repo_id']}")
            continue
    
    # Summary after download completion
    logger.info(f"\n{'='*50}")
    logger.info("Download task completed!")
    logger.info(f"Successfully downloaded: {success_count}/{len(MODELS)} models")
    
    if failed_models:
        logger.warning(f"Failed models: {', '.join(failed_models)}")
    else:
        logger.info("🎉 All models downloaded successfully!")
    
    logger.info("\nModel save locations:")
    for model in MODELS:
        if model["repo_id"] not in failed_models:
            logger.info(f"  - {model['repo_id']} -> {model['local_dir']}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\nDownload interrupted by user")
    except Exception as e:
        logger.error(f"Program execution error: {str(e)}")