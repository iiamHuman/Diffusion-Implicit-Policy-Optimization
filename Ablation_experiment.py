import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
from datasets import load_dataset
from transformers import AutoProcessor, AutoModel, CLIPProcessor, CLIPModel
from diffusers import StableDiffusionXLPipeline, DDIMScheduler
from tqdm import tqdm
from torchvision.transforms import ToTensor
import torch.multiprocessing as mp
from torch.cuda.amp import autocast
from multiprocessing import Process, Queue, Manager

class Config:
    prompt_count = 1632
    ddim_steps = 30
    eta_values = [0.0, 0.2, 0.5, 0.7, 1.0]
    model_path = "/your/model/path/"
    output_dir = "./Ablation"
    devices = ["cuda:4", "cuda:5"]
    pickscore_model = "./PickScore_v1/AI-ModelScope/PickScore_v1"
    pickscore_processor = "./CLIP-ViT-H-14-laion2B-s32B-b79K"
    hps_model = "./HPSv2-hf"
    hps_processor = "./clip-vit-base-patch32"


def load_prompts(n):
    ds = load_dataset("./nateraw", split="train")
    return ds["Prompt"][:n]


def truncate_text(processor, text, max_length=77):
    tokens = processor.tokenizer(text, return_tensors="pt", truncation=False)
    if tokens.input_ids.shape[1] > max_length:
        truncated_tokens = processor.tokenizer(
            text, 
            return_tensors="pt", 
            truncation=True, 
            max_length=max_length,
            padding=True
        )
        return processor.tokenizer.decode(truncated_tokens.input_ids[0], skip_special_tokens=True)
    return text


def load_image_models(device):
    processor = AutoProcessor.from_pretrained(Config.pickscore_processor)
    model = AutoModel.from_pretrained(Config.pickscore_model).to(device).eval()
    return processor, model

def load_hps_model(device):
    processor = CLIPProcessor.from_pretrained(Config.hps_processor)
    model = CLIPModel.from_pretrained(Config.hps_model).to(device).eval()
    return processor, model

def load_clipscore_model(device):
    processor = CLIPProcessor.from_pretrained("./clip-vit-base-patch32")
    model = CLIPModel.from_pretrained("./clip-vit-base-patch32").to(device).eval()
    return processor, model


def calc_pickscore(prompt, image, processor, model, device):
    truncated_prompt = truncate_text(processor, prompt)
    
    inputs = processor(images=image, return_tensors="pt").to(device)
    text_inputs = processor(text=truncated_prompt, return_tensors="pt", truncation=True, max_length=77).to(device)
    
    with torch.no_grad():
        image_emb = model.get_image_features(**inputs)
        text_emb = model.get_text_features(**text_inputs)
        image_emb = image_emb / image_emb.norm(dim=-1, keepdim=True)
        text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)
        score = model.logit_scale.exp() * (text_emb @ image_emb.T)
    return score.item()

def calc_hps(prompt, image, processor, model, device):
    truncated_prompt = truncate_text(processor, prompt)
    
    inputs = processor(
        images=image, 
        text=truncated_prompt, 
        return_tensors="pt", 
        padding=True,
        truncation=True,
        max_length=77
    ).to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits_per_image
    return logits.item()

def calc_clipscore(prompt, image, processor, model, device):
    truncated_prompt = truncate_text(processor, prompt)
    
    text_inputs = processor(
        text=truncated_prompt, 
        return_tensors="pt", 
        padding=True,
        truncation=True,
        max_length=77
    ).to(device)
    
    image_inputs = processor(images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        text_emb = model.get_text_features(**text_inputs)
        image_emb = model.get_image_features(**image_inputs)

    text_emb = text_emb / text_emb.norm(p=2, dim=-1, keepdim=True)
    image_emb = image_emb / image_emb.norm(p=2, dim=-1, keepdim=True)

    score = (text_emb @ image_emb.T).squeeze().item() * 100
    return score


def load_pipeline(device):
    pipe = StableDiffusionXLPipeline.from_pretrained(
        Config.model_path, torch_dtype=torch.float16
    ).to(device)
    return pipe

def generate_images(pipe, prompt, eta):
    scheduler = DDIMScheduler.from_pretrained(Config.model_path, subfolder="scheduler")
    scheduler.set_timesteps(Config.ddim_steps)
    scheduler.eta = eta

    pipe.scheduler = scheduler
    
    image = pipe(prompt, num_inference_steps=Config.ddim_steps, guidance_scale=7.5).images[0]
    return image


def worker(device, prompts_chunk, eta, result_queue):

    torch.cuda.set_device(device)
    
    pipe = load_pipeline(device)
    
    pick_processor, pick_model = load_image_models(device)
    hps_processor, hps_model = load_hps_model(device)
    clipscore_processor, clipscore_model = load_clipscore_model(device)
    
    device_results = []
    
    for idx, prompt in enumerate(prompts_chunk):
        try:
            image = generate_images(pipe, prompt, eta)
            
            pick = calc_pickscore(prompt, image, pick_processor, pick_model, device)
            hps = calc_hps(prompt, image, hps_processor, hps_model, device)
            clip = calc_clipscore(prompt, image, clipscore_processor, clipscore_model, device)
            
            image_path = os.path.join(Config.output_dir, f"eta{eta}_idx{idx}_dev{device.split(':')[-1]}.png")
            image.save(image_path)
            
            device_results.append({
                "eta": eta,
                "prompt": prompt,
                "pick_score": pick,
                "hps_score": hps,
                "clip_score": clip,
                "image_path": image_path
            })
            
        except Exception as e:
            continue
    
    result_queue.put(device_results)

def evaluate():
    os.makedirs(Config.output_dir, exist_ok=True)
    prompts = load_prompts(Config.prompt_count)
    
    pick_processor, _ = load_image_models(Config.devices[0])
    long_prompts = []
    for i, prompt in enumerate(prompts):
        tokens = pick_processor.tokenizer(prompt, return_tensors="pt", truncation=False)
        if tokens.input_ids.shape[1] > 77:
            long_prompts.append((i, prompt, tokens.input_ids.shape[1]))
    
    if long_prompts:
        manager = Manager()
        results = manager.dict()

    for eta in Config.eta_values:

        chunks = np.array_split(prompts, len(Config.devices))
        
        processes = []
        result_queue = Queue()
        
        for i, device in enumerate(Config.devices):
            chunk = chunks[i].tolist()
            p = Process(
                target=worker,
                args=(device, chunk, eta, result_queue))
            p.start()
            processes.append(p)
        
        all_results = []
        for _ in range(len(Config.devices)):
            all_results.extend(result_queue.get())
        

        for p in processes:
            p.join()
        
        results[str(eta)] = all_results
    
    final_results = {"eta": [], "steps": [], "PickScore": [], "HPSv2": [], "CLIPScore": []}
    
    for eta in Config.eta_values:
        eta_results = results[str(eta)]
        
        if not eta_results:
            continue
            
        pick_scores = [r["pick_score"] for r in eta_results]
        hps_scores = [r["hps_score"] for r in eta_results]
        clip_scores = [r["clip_score"] for r in eta_results]
        
        final_results["eta"].append(str(eta))
        final_results["steps"].append(Config.ddim_steps)
        final_results["PickScore"].append(np.mean(pick_scores))
        final_results["HPSv2"].append(np.mean(hps_scores))
        final_results["CLIPScore"].append(np.mean(clip_scores))
    
    detailed_results = []
    for eta in Config.eta_values:
        detailed_results.extend(results[str(eta)])
    
    detailed_df = pd.DataFrame(detailed_results)
    detailed_df.to_csv(os.path.join(Config.output_dir, "detailed_results.csv"), index=False)
    
    df = pd.DataFrame(final_results)
    df.to_csv(os.path.join(Config.output_dir, "summary_results.csv"), index=False)
    
    return df


def visualize(df):
    plt.figure(figsize=(10, 5))
    bar_width = 0.25
    x = np.arange(len(df["eta"]))
    plt.bar(x - bar_width, df["PickScore"], width=bar_width, label="PickScore")
    plt.bar(x, df["HPSv2"], width=bar_width, label="HPSv2")
    plt.bar(x + bar_width, df["CLIPScore"], width=bar_width, label="CLIPScore")
    for i, (p, h, c) in enumerate(zip(df["PickScore"], df["HPSv2"], df["CLIPScore"])):
        plt.text(i - 0.28, p + 0.01, f"{p:.2f}", fontsize=8)
        plt.text(i - 0.02, h + 0.01, f"{h:.2f}", fontsize=8)
        plt.text(i + 0.22, c + 0.01, f"{c:.2f}", fontsize=8)
    plt.xticks(x, df["eta"])
    plt.ylabel("Score")
    plt.xlabel("η")
    plt.title(f"DIPO η vs PickScore / HPSv2 / CLIPScore (steps={Config.ddim_steps})")
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=3)
    plt.tight_layout()
    plt.savefig(os.path.join(Config.output_dir, "bar_plot.png"))
    plt.close()

    fig, ax = plt.subplots()
    table_data = list(zip(
        df["eta"], df["steps"], df["PickScore"].round(2), df["HPSv2"].round(2), df["CLIPScore"].round(2)
    ))
    col_labels = ["η", "Steps", "PickScore", "HPSv2", "CLIPScore"]
    ax.axis("off")
    table = ax.table(cellText=table_data, colLabels=col_labels, loc="center")
    table.scale(1.2, 1.5)
    plt.savefig(os.path.join(Config.output_dir, "table_plot.png"))
    plt.close()

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    
    df = evaluate()
    visualize(df)
