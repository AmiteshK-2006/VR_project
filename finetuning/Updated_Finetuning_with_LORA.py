import os
import glob
import json
import time
from datetime import timedelta, datetime
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BlipForQuestionAnswering, BlipProcessor, get_linear_schedule_with_warmup
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from accelerate import Accelerator
from tqdm.auto import tqdm
import numpy as np
import psutil
import gc
import warnings
from torch.cuda.amp import autocast
from torch.utils.data.distributed import DistributedSampler
import random
import bitsandbytes as bnb
from peft import TaskType
import types

warnings.filterwarnings("ignore")

print("Imports Successful")

# Configuration
class Config:
    # Paths
    TRAIN_JSON_DIR = "./lora-finetuning"
    VAL_JSON_DIR = "./validation-dataset/validation"
    IMAGE_PREFIX = "./berkley-dataset"
    PRETRAINED_DIR = "./pretrained-NAHICHALTA"
    CHECKPOINT_DIR = "./working/checkpoints"
    LOGS_DIR = "./working/logs"

    # Training parameters
    BATCH_SIZE = 8
    NUM_WORKERS = 2
    NUM_EPOCHS = 3
    LEARNING_RATE = 2e-4
    WARMUP_RATIO = 0.1
    WEIGHT_DECAY = 0.01
    GRADIENT_ACCUMULATION_STEPS = 4

    # LoRA parameters
    LORA_R = 4
    LORA_ALPHA = 8
    LORA_DROPOUT = 0.1
    TARGET_MODULES = ["q_proj", "v_proj", "k_proj", "out_proj", "fc1", "fc2"]

    # Data preprocessing
    IMAGE_SIZE = 384
    MAX_LENGTH = 32

    # Optimization and logging
    FP16 = True
    LOG_EVERY = 100
    MIXED_PRECISION = None

    # Flash attention 
    USE_FLASH_ATTENTION = True

    # 8-bit quantization for base model
    USE_8BIT_QUANT = True

    # Reproducibility
    SEED = 42

    # Training optimizations
    USE_GRADIENT_CHECKPOINTING = True
    USE_DISTRIBUTED_TRAINING = True
    USE_AMP = False
    PREFETCH_FACTOR = 2

    # Dataset sampling
    NUM_SAMPLES = None

    # New checkpoint parameters
    CHECKPOINT_EVERY_N_IMAGES = 2000  # Save every 2000 images (85,000 QA pairs)
    IMAGES_PER_QA_PAIR = 1/15  # Each QA pair represents 1/15 of an image

# Create directories
config = Config()
os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
os.makedirs(config.LOGS_DIR, exist_ok=True)

print("Configurations learned successfully")

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

set_seed(config.SEED)

print("Seed is set")

def log_info(message):
    """Log information to console and file"""
    print(f"[INFO] {message}")
    with open(os.path.join(config.LOGS_DIR, "training_log.txt"), "a") as f:
        f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}\n")

def save_checkpoint_metadata(checkpoint_path, epoch, global_step, datapoints_processed,
                           images_processed, train_loss, val_loss=None, lr=None):
    """Save metadata about the checkpoint to a JSON file"""
    metadata = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "checkpoint_path": checkpoint_path,
        "epoch": epoch + 1,
        "global_step": global_step,
        "datapoints_processed": datapoints_processed,
        "images_processed": images_processed,
        "train_loss": train_loss,
        "learning_rate": lr,
        "val_loss": val_loss,
    }

    metadata_path = os.path.join(os.path.dirname(checkpoint_path),
                                 os.path.basename(checkpoint_path) + "_metadata.json")

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    log_info(f"Checkpoint metadata saved to {metadata_path}")
    return metadata_path

log_info(f"Starting training at {time.strftime('%Y-%m-%d %H:%M:%S')}")
log_info(f"Configuration: {vars(config)}")

if torch.cuda.is_available():
    log_info(f"GPU: {torch.cuda.get_device_name(0)}")
    log_info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    log_info(f"CUDA Version: {torch.version.cuda}")

pretrained_ckpts = glob.glob(os.path.join(config.PRETRAINED_DIR, "*"))
if pretrained_ckpts:
    log_info(f"Finetuning from pretrained checkpoint: {pretrained_ckpts[0]}")
    base_model_path = pretrained_ckpts[0]
else:
    log_info("No pretrained found. Initializing new model from 'blip-vqa-base'")
    base_model_path = "Salesforce/blip-vqa-base"

print("Model Creation Successful")

class VQADataset(Dataset):
    def __init__(self, json_dir, processor, max_length=config.MAX_LENGTH,
                 image_size=config.IMAGE_SIZE, num_samples=config.NUM_SAMPLES,
                 is_training=True, cache_images=False):
        self.processor = processor
        self.max_length = max_length
        self.image_size = image_size
        self.samples = []
        self.cache_images = cache_images
        self.image_cache = {}
        self.is_training = is_training
        self.unique_image_paths = set()  # Track unique images for counting

        log_info(f"Loading samples from {json_dir}...")
        json_files = glob.glob(os.path.join(json_dir, "*.json"))

        for fp in tqdm(json_files, desc="Loading dataset"):
            try:
                # Skip empty files
                if os.path.getsize(fp) == 0:
                    log_info(f"Skipping empty file {fp}")
                    continue

                with open(fp, 'r') as f:
                    data = json.load(f)

                # Validate top-level structure
                if not isinstance(data, list) or len(data) != 2:
                    raise ValueError(f"Expected list of length 2, got {type(data)} with length {len(data)}")

                image_path_data, questions_data = data

                # Validate required keys
                if "path" not in image_path_data or "questions" not in questions_data:
                    raise ValueError(f"Missing 'path' or 'questions' keys")

                image_path = os.path.join(config.IMAGE_PREFIX, image_path_data["path"])

                # Add to unique images set
                self.unique_image_paths.add(image_path)

                # Extract QA pairs
                for qa in questions_data["questions"]:
                    if isinstance(qa, dict) and "question" in qa and "answer" in qa:
                        self.samples.append((image_path, qa["question"], qa["answer"]))
                    else:
                        raise ValueError(f"Malformed QA entry: {qa}")

            except json.JSONDecodeError as jde:
                log_info(f"JSON decode error in {fp}: {jde}")
            except Exception as e:
                log_info(f"Error loading {fp}: {e}")

        # Optional subsample for speed
        if num_samples and num_samples < len(self.samples):
            random.shuffle(self.samples)
            self.samples = self.samples[:num_samples]
            # Recalculate unique image paths based on subsampled data
            self.unique_image_paths = set(img_path for img_path, _, _ in self.samples)
            log_info(f"Subsampled to {num_samples} samples")

        self.num_unique_images = len(self.unique_image_paths)
        log_info(f"Loaded {len(self.samples)} QA pairs from {self.num_unique_images} unique images")


    def __len__(self):
        return len(self.samples)

    def preprocess_image(self, image_path):
        image = Image.open(image_path).convert("RGB")

        if self.is_training:
            # Simple data augmentation for training
            if random.random() > 0.5:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)

        return image

    def __getitem__(self, idx):
        img_path, question, answer = self.samples[idx]
        try:
            # Process image
            image = self.preprocess_image(img_path)

            # Process question with processor
            encoded_question = self.processor(
                text=question,
                images=image,
                return_tensors="pt"
            )

            # Process answer with consistent padding
            encoded_answer = self.processor.tokenizer(
                answer,
                padding="max_length",
                max_length=self.max_length,
                truncation=True,
                return_tensors="pt"
            )

            # Flatten batch dimension
            inputs = {
                "input_ids": encoded_question.input_ids.squeeze(),
                "attention_mask": encoded_question.attention_mask.squeeze(),
                "pixel_values": encoded_question.pixel_values.squeeze(),
                "labels": encoded_answer.input_ids.squeeze()
            }

            return inputs

        except Exception as e:
            print(f"Error processing sample {idx}: {str(e)}")
            return self._get_dummy_sample()

    def _get_dummy_sample(self):
        dummy = {}
        dummy["input_ids"] = torch.zeros(1, dtype=torch.long)
        dummy["attention_mask"] = torch.zeros(1, dtype=torch.long)
        dummy["pixel_values"] = torch.zeros((3, self.image_size, self.image_size), dtype=torch.float)
        dummy["labels"] = torch.zeros(1, dtype=torch.long)
        return dummy

print("Class for dataset defined")

def collate_fn(batch):
    valid_batch = [item for item in batch if item["input_ids"].numel() > 0 and item["pixel_values"].shape[0] == 3]

    if not valid_batch:
        return dummy_batch()

    # Stack pixel values
    pixel_values = torch.stack([item["pixel_values"] for item in valid_batch])

    # Determine max lengths
    max_q_len = max([item["input_ids"].shape[0] for item in valid_batch])
    max_a_len = max([item["labels"].shape[0] for item in valid_batch])

    batch_size = len(valid_batch)
    pad_token_id = processor.tokenizer.pad_token_id

    # Initialize tensors
    input_ids = torch.full((batch_size, max_q_len), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((batch_size, max_q_len), dtype=torch.long)
    labels = torch.full((batch_size, max_a_len), pad_token_id, dtype=torch.long)

    # Fill tensors
    for i, item in enumerate(valid_batch):
        q_len = item["input_ids"].shape[0]
        input_ids[i, :q_len] = item["input_ids"]
        attention_mask[i, :q_len] = item["attention_mask"]

        a_len = item["labels"].shape[0]
        labels[i, :a_len] = item["labels"]

    # Create batch
    batch_dict = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "pixel_values": pixel_values,
        "labels": labels
    }

    return batch_dict

def dummy_batch():
    # Create a minimal dummy batch when collation fails
    batch_dict = {
        "input_ids": torch.zeros((1, 1), dtype=torch.long),
        "attention_mask": torch.zeros((1, 1), dtype=torch.long),
        "pixel_values": torch.zeros((1, 3, config.IMAGE_SIZE, config.IMAGE_SIZE), dtype=torch.float),
        "labels": torch.zeros((1, 1), dtype=torch.long)
    }
    return batch_dict

def print_model_size(model):
    param_size = 0
    buffer_size = 0

    for param in model.parameters():
        param_size += param.nelement() * param.element_size()

    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_mb = (param_size + buffer_size) / 1024**2
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())

    log_info(f"Model size: {size_mb:.2f} MB")
    log_info(f"Total parameters: {total_params:,}")
    log_info(f"Trainable parameters: {trainable_params:,} ({trainable_params/total_params*100:.2f}%)")

def print_memory_usage():
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    memory_mb = memory_info.rss / (1024 * 1024)

    log_info(f"System Memory usage: {memory_mb:.2f} MB")

    if torch.cuda.is_available():
        log_info(f"CUDA Memory - Allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        log_info(f"CUDA Memory - Reserved: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
        log_info(f"CUDA Memory - Max Allocated: {torch.cuda.max_memory_allocated() / 1024**2:.2f} MB")

print("Utilities Function Registered")

def setup_flashattention(model):
    try:
        from flash_attn.ops.fused_dense import FusedDense
        from flash_attn.modules.mha import FlashSelfAttention

        log_info("Setting up Flash Attention...")

        # Replace attention layers with flash attention
        count = 0
        for name, module in model.named_modules():
            if "attention" in name.lower() and hasattr(module, "query"):
                # Set flash attention flag if available in the model
                if hasattr(module, "use_flash_attention"):
                    module.use_flash_attention = True
                    count += 1

        log_info(f"Enabled Flash Attention for {count} modules")
        return True
    except ImportError:
        log_info("Flash Attention not available. Continuing with standard attention.")
        return False

def get_kvcache_model(model):
    try:
        model.config.use_cache = True
        log_info("Enabled KV-Cache optimization")
        return True
    except:
        log_info("KV-Cache optimization not applied")
        return False

def apply_gradient_checkpointing(model):
    try:
        model.gradient_checkpointing_enable()
        log_info("Gradient checkpointing enabled")
        return True
    except:
        log_info("Could not enable gradient checkpointing")
        return False

print("Enhancement Functions registered")

def fix_blip_for_training(model):
    """Comprehensive fix for BLIP model to address both in-place operations and batch size issues."""
    print("Applying comprehensive BLIP model fixes...")

    # 1. Fix in-place operations in embeddings
    if hasattr(model, 'base_model') and hasattr(model.base_model, 'text_encoder'):
        text_encoder = model.base_model.text_encoder
        if hasattr(text_encoder, 'embeddings'):
            # Get the original embeddings forward method
            original_embeddings_forward = text_encoder.embeddings.forward

            # Define a non-in-place version
            def safe_embeddings_forward(self, input_ids=None, position_ids=None, inputs_embeds=None, past_key_values_length=0):
                if input_ids is not None:
                    input_shape = input_ids.size()
                else:
                    input_shape = inputs_embeds.size()[:-1]

                seq_length = input_shape[1]

                if position_ids is None:
                    position_ids = torch.arange(
                        past_key_values_length, seq_length + past_key_values_length,
                        dtype=torch.long, device=input_ids.device if input_ids is not None else inputs_embeds.device
                    )
                    position_ids = position_ids.unsqueeze(0).expand(input_shape)

                if inputs_embeds is None:
                    inputs_embeds = self.word_embeddings(input_ids)

                # NON-IN-PLACE VERSION: Use + operator instead of +=
                embeddings = inputs_embeds.clone()

                if self.position_embedding_type == "absolute":
                    position_embeddings = self.position_embeddings(position_ids)
                    # Create a new tensor with addition instead of modifying in-place
                    embeddings = embeddings + position_embeddings

                embeddings = self.LayerNorm(embeddings)
                embeddings = self.dropout(embeddings)

                return embeddings

            # Replace the embeddings forward method
            text_encoder.embeddings.forward = types.MethodType(safe_embeddings_forward, text_encoder.embeddings)
            print("✅ Fixed in-place operations in BLIP embeddings")

    # 2. Fix forward pass to handle proper decoder_input_ids and batch size issues
    original_forward = model.forward

    def new_forward(self, **kwargs):
        # Only include needed inputs
        filtered_kwargs = {}
        for k, v in kwargs.items():
            if k in ["input_ids", "attention_mask", "pixel_values", "labels"]:
                # Create clone to prevent in-place modifications
                if isinstance(v, torch.Tensor):
                    filtered_kwargs[k] = v.clone()
                else:
                    filtered_kwargs[k] = v

        # Generate decoder_input_ids from labels
        if 'labels' in filtered_kwargs:
            labels = filtered_kwargs['labels']
            batch_size, seq_len = labels.shape

            # Create decoder_input_ids: [BOS, label[:-1]]
            pad_token_id = self.base_model.config.text_config.pad_token_id
            bos_token_id = getattr(self.base_model.config.text_config, 'bos_token_id',
                                   self.base_model.config.text_config.decoder_start_token_id)

            # Create decoder_input_ids
            decoder_input_ids = torch.full_like(labels, pad_token_id)
            decoder_input_ids[:, 0] = bos_token_id  # First token is BOS

            # Shift labels to right (only if sequence length > 1)
            if seq_len > 1:
                decoder_input_ids[:, 1:] = labels[:, :-1].clone()

            # Create decoder attention mask
            decoder_attention_mask = (decoder_input_ids != pad_token_id).long()

            # Add to kwargs
            filtered_kwargs['decoder_input_ids'] = decoder_input_ids
            filtered_kwargs['decoder_attention_mask'] = decoder_attention_mask

            # Special handling for ignore_index (-100)
            # Replace -100 with pad token for consistent shape handling
            valid_label_mask = labels != -100
            if not valid_label_mask.all():
                new_labels = labels.clone()
                new_labels[~valid_label_mask] = pad_token_id
                filtered_kwargs['labels'] = new_labels

        # Call base model with fixed inputs
        if hasattr(self, 'base_model'):
            if not hasattr(self, '_debug_printed'):
                print("First batch shapes:")
                for k, v in filtered_kwargs.items():
                    if isinstance(v, torch.Tensor):
                        print(f"  {k}: {v.shape}")
                self._debug_printed = True

            # Try to run the model
            try:
                return self.base_model(**filtered_kwargs)
            except Exception as e:
                print(f"Error in forward pass: {str(e)}")
                print("Input shapes:")
                for k, v in filtered_kwargs.items():
                    if isinstance(v, torch.Tensor):
                        print(f"  {k}: {v.shape}")
                        print(f"  {k} unique values: {torch.unique(v).tolist()[:10]}...")
                raise
        else:
            return original_forward(**filtered_kwargs)

    # Replace the model's forward method
    model.forward = types.MethodType(new_forward, model)
    print("✅ Fixed model forward method for batch size consistency")

    return model

setup_start_time = time.time()

processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")

loading_kwargs = {}
if config.USE_8BIT_QUANT and torch.cuda.is_available():
    try:
        log_info("Loading model in 8-bit precision")
        loading_kwargs.update({
            "load_in_8bit": True,
            "device_map": "auto",
        })
    except:
        log_info("8-bit loading failed, falling back to full precision")

model = BlipForQuestionAnswering.from_pretrained(base_model_path)

if config.USE_FLASH_ATTENTION:
    flash_avail = setup_flashattention(model)
kv_cache_enabled = get_kvcache_model(model)

print_model_size(model)

if config.USE_GRADIENT_CHECKPOINTING:
    gc_enabled = apply_gradient_checkpointing(model)

if config.USE_8BIT_QUANT and "load_in_8bit" in loading_kwargs and loading_kwargs["load_in_8bit"]:
    try:
        model = prepare_model_for_kbit_training(model)
        log_info("Model prepared for k-bit training")
    except Exception as e:
        log_info(f"Error preparing model for k-bit training: {str(e)}")

# Initialize Accelerator with optimal settings
accelerator = Accelerator(
    mixed_precision=config.MIXED_PRECISION,
    gradient_accumulation_steps=config.GRADIENT_ACCUMULATION_STEPS,
    log_with=None,
    cpu=False if torch.cuda.is_available() else True,
)

device = accelerator.device
log_info(f"Using device: {device}")
log_info(f"Distributed training: {accelerator.distributed_type}")
log_info(f"Mixed precision: {config.MIXED_PRECISION}")

from peft import get_peft_model, LoraConfig, TaskType

# Define simplified LoRA config
lora_config = LoraConfig(
    r=4,  # Lower rank
    lora_alpha=8,  # Lower alpha
    lora_dropout=0.05,
    target_modules=["query", "value", "key"],  # Target only core modules
    task_type=TaskType.SEQ_2_SEQ_LM,
    bias="none",
    inference_mode=False
)

# Apply LoRA
model = get_peft_model(model, lora_config)
print("LoRA applied")

# Apply comprehensive fixes
model = fix_blip_for_training(model)
print("Model prepared for training")

def print_trainable_parameters(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable} / {total} ({100 * trainable / total:.2f}%)")

print("✅ LoRA applied to BLIP model.")
print_trainable_parameters(model)

train_ds = VQADataset(
    config.TRAIN_JSON_DIR,
    processor,
    is_training=True,
    cache_images=False
)

val_ds = VQADataset(
    config.VAL_JSON_DIR,
    processor,
    is_training=False,
    cache_images=True
)

log_info(f"Training dataset: {len(train_ds)} samples")
log_info(f"Validation dataset: {len(val_ds)} samples")
log_info(f"Number of unique images in training dataset: {train_ds.num_unique_images}")

train_sampler = DistributedSampler(train_ds) if accelerator.num_processes > 1 else None
val_sampler = DistributedSampler(val_ds, shuffle=False) if accelerator.num_processes > 1 else None

train_loader = DataLoader(
    train_ds,
    batch_size=config.BATCH_SIZE,
    shuffle=(train_sampler is None),
    sampler=train_sampler,
    num_workers=config.NUM_WORKERS,
    pin_memory=True,
    prefetch_factor=config.PREFETCH_FACTOR if hasattr(torch.utils.data, 'prefetch_factor') else 2,
    persistent_workers=True if config.NUM_WORKERS > 0 else False,
    collate_fn=collate_fn,
    drop_last=True
)
val_loader = DataLoader(
    val_ds,
    batch_size=config.BATCH_SIZE,
    shuffle=False,
    sampler=val_sampler,
    num_workers=config.NUM_WORKERS,
    pin_memory=True,
    collate_fn=collate_fn
)

if hasattr(bnb, "optim") and hasattr(bnb.optim, "AdamW8bit"):
    use_8bit_optimizer = True
    log_info("Using 8-bit AdamW optimizer")
else:
    use_8bit_optimizer = False
    log_info("Using standard AdamW optimizer")

no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
optimizer_grouped_parameters = [
    {
        "params": [p for n, p in model.named_parameters()
                  if p.requires_grad and not any(nd in n for nd in no_decay)],
        "weight_decay": config.WEIGHT_DECAY,
    },
    {
        "params": [p for n, p in model.named_parameters()
                  if p.requires_grad and any(nd in n for nd in no_decay)],
        "weight_decay": 0.0,
    },
]

if use_8bit_optimizer:
    try:
        optimizer = bnb.optim.AdamW8bit(
            optimizer_grouped_parameters,
            lr=config.LEARNING_RATE,
            betas=(0.9, 0.999),
            eps=1e-8
        )
    except:
        log_info("8-bit optimizer failed, falling back to standard AdamW")
        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=config.LEARNING_RATE
        )
else:
    optimizer = torch.optim.AdamW(
        optimizer_grouped_parameters,
        lr=config.LEARNING_RATE
    )

total_steps = len(train_loader) * config.NUM_EPOCHS // accelerator.gradient_accumulation_steps
warmup_steps = int(total_steps * config.WARMUP_RATIO)

lr_scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_steps
)

setup_time = time.time() - setup_start_time
log_info(f"Setup completed in {timedelta(seconds=int(setup_time))}")

model, optimizer, train_loader, val_loader, lr_scheduler = accelerator.prepare(
    model, optimizer, train_loader, val_loader, lr_scheduler
)

print("Model Setup Complete")

# Initialize counters
global_step = 0
best_val_loss = float('inf')
start_time = time.time()
datapoints_processed = 0
images_processed = 0
last_checkpoint_datapoints = 0
last_checkpoint_images = 0

# Training loop with saving by data points
try:
    for epoch in range(config.NUM_EPOCHS):
        epoch_start = time.time()

        # Set sampler epoch if using distributed training
        # if train_sampler is not None:
        #     train_sampler.set_epoch(epoch)

        # Training phase
        model.train()
        train_loss = torch.tensor(0.0, device=accelerator.device)

        # Setup progress bar
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader),
                          desc=f"Epoch {epoch+1}/{config.NUM_EPOCHS}")

        for step, batch in progress_bar:
            # Forward pass
            outputs = model(**batch)
            loss = outputs.loss

            # Backward pass (non-in-place)
            scaled_loss = loss / accelerator.gradient_accumulation_steps
            accelerator.backward(scaled_loss)

            # Track loss without in-place operations
            train_loss = train_loss + loss.detach()

            # Update processed data counts
            batch_size = batch["input_ids"].shape[0]
            datapoints_processed += batch_size
            images_processed += batch_size * config.IMAGES_PER_QA_PAIR

            # Optimizer step
            if (step + 1) % accelerator.gradient_accumulation_steps == 0 or step == len(train_loader) - 1:
                accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                global_step += 1

                # Update progress bar
                progress_bar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "images": f"{int(images_processed)}",
                    "datapoints": f"{datapoints_processed}"
                })

                # Log information periodically
                if global_step % config.LOG_EVERY == 0:
                    current_lr = lr_scheduler.get_last_lr()[0] if lr_scheduler else config.LEARNING_RATE
                    elapsed = time.time() - start_time
                    log_info(
                        f"Step {global_step} | "
                        f"Loss: {loss.item():.4f} | "
                        f"LR: {current_lr:.7f} | "
                        f"Images: {int(images_processed)} | "
                        f"Data points: {datapoints_processed} | "
                        f"Time: {timedelta(seconds=int(elapsed))}"
                    )

            # Check for checkpoint based on images processed
            if accelerator.is_main_process:
                images_since_last_checkpoint = images_processed - last_checkpoint_images
                if images_since_last_checkpoint >= config.CHECKPOINT_EVERY_N_IMAGES:
                    # Save checkpoint
                    checkpoint_path = os.path.join(config.CHECKPOINT_DIR, f"checkpoint-{int(images_processed)}-images")
                    unwrapped_model = accelerator.unwrap_model(model)
                    unwrapped_model.save_pretrained(checkpoint_path)

                    # Save metadata
                    avg_train_loss = train_loss.item() / (step + 1)
                    current_lr = lr_scheduler.get_last_lr()[0] if lr_scheduler else config.LEARNING_RATE

                    save_checkpoint_metadata(
                        checkpoint_path=checkpoint_path,
                        epoch=epoch,
                        global_step=global_step,
                        datapoints_processed=datapoints_processed,
                        images_processed=int(images_processed),
                        train_loss=avg_train_loss,
                        lr=current_lr
                    )

                    # Update checkpoint counters
                    last_checkpoint_images = images_processed
                    last_checkpoint_datapoints = datapoints_processed

                    log_info(f"Checkpoint saved at {int(images_processed)} images / {datapoints_processed} datapoints")

        # Calculate average train loss for this epoch
        avg_train_loss = train_loss.item() / len(train_loader)
        print(f"Epoch {epoch+1} completed, Avg. train loss: {avg_train_loss:.4f}")
        # Validation phase
        model.eval()
        val_loss = torch.tensor(0.0, device=accelerator.device)

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                outputs = model(**batch)
                val_loss = val_loss + outputs.loss

        avg_val_loss = val_loss.item() / len(val_loader)
        print(f"Validation loss: {avg_val_loss:.4f}")
        # Log epoch results
        epoch_time = time.time() - epoch_start
        log_info(
            f"Epoch {epoch+1}/{config.NUM_EPOCHS} completed in {timedelta(seconds=int(epoch_time))} | "
            f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | "
            f"Images: {int(images_processed)} | Datapoints: {datapoints_processed}"
        )

        # Save epoch checkpoint
        if accelerator.is_main_process:
            epoch_checkpoint_path = os.path.join(config.CHECKPOINT_DIR, f"checkpoint-epoch-{epoch+1}")
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(epoch_checkpoint_path)

            # Save metadata for epoch checkpoint
            current_lr = lr_scheduler.get_last_lr()[0] if lr_scheduler else config.LEARNING_RATE

            save_checkpoint_metadata(
                checkpoint_path=epoch_checkpoint_path,
                epoch=epoch,
                global_step=global_step,
                datapoints_processed=datapoints_processed,
                images_processed=int(images_processed),
                train_loss=avg_train_loss,
                val_loss=avg_val_loss,
                lr=current_lr
            )

            # Save best model if val loss improved
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_checkpoint_path = os.path.join(config.CHECKPOINT_DIR, "best_model")
                unwrapped_model.save_pretrained(best_checkpoint_path)

                save_checkpoint_metadata(
                    checkpoint_path=best_checkpoint_path,
                    epoch=epoch,
                    global_step=global_step,
                    datapoints_processed=datapoints_processed,
                    images_processed=int(images_processed),
                    train_loss=avg_train_loss,
                    val_loss=avg_val_loss,
                    lr=current_lr
                )

                log_info(f"New best model saved with validation loss: {best_val_loss:.4f}")

        # Memory cleanup
        gc.collect()
        torch.cuda.empty_cache()

    # Training complete
    total_time = time.time() - start_time
    log_info(f"Training completed in {timedelta(seconds=int(total_time))}")
    log_info(f"Processed {datapoints_processed} datapoints / {int(images_processed)} images")
    log_info(f"Best validation loss: {best_val_loss:.4f}")

except Exception as e:
    log_info(f"Training failed: {str(e)}")
    import traceback
    traceback.print_exc()
finally:
    # Final cleanup
    accelerator.free_memory()
    log_info("Training script finished.")
