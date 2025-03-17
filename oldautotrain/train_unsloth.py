import torch
from unsloth import FastModel
from datasets import load_dataset
from unsloth.chat_templates import get_chat_template, standardize_data_formats, train_on_responses_only
from trl import SFTTrainer, SFTConfig

import json  # For better debug output

# ✅ Load the optimized Unsloth model
model_name = "unsloth/gemma-3-4b-it-unsloth-bnb-4bit"
model, tokenizer = FastModel.from_pretrained(
    model_name=model_name,
    max_seq_length=2048,
    load_in_4bit=True,  # Optimized for efficiency
    load_in_8bit=False,
    full_finetuning=False,
    attn_implementation='eager'
)

# ✅ Enable LoRA fine-tuning
model = FastModel.get_peft_model(
    model,
    finetune_vision_layers=False,
    finetune_language_layers=True,
    finetune_attention_modules=True,
    finetune_mlp_modules=True,
    r=8,
    lora_alpha=8,
    lora_dropout=0,
    bias="none",
    random_state=3407,
)

# ✅ Apply the correct chat template
tokenizer = get_chat_template(tokenizer, chat_template="gemma-3")

# ✅ Load your custom JSONL dataset
dataset = load_dataset("json", data_files="train.jsonl", split="train")

# ✅ Standardize dataset format
dataset = standardize_data_formats(dataset)

def apply_chat_template(examples):
    formatted_examples = []
    examples_dict = {key: list(value) for key, value in examples.items()}
    
    for inp, out in zip(examples_dict["input"], examples_dict["output"]):
        conversation = [
            {"role": "user", "content": inp.strip()},  
            {"role": "assistant", "content": out.strip()}
        ]
        try:
            formatted_text = tokenizer.apply_chat_template(conversation)
            formatted_examples.append(formatted_text)
        except Exception as e:
            print("\n❌ ERROR: Failed to format chat template!")
            print(json.dumps(conversation, indent=2, ensure_ascii=False))
            raise e
    
    return {"text": formatted_examples}

# ✅ Apply chat template transformation
dataset = dataset.map(apply_chat_template, batched=True)

# ✅ Set up the fine-tuning trainer
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    eval_dataset=None,
    args=SFTConfig(
        dataset_text_field="text",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        max_steps=10000,
        learning_rate=2e-4,
        logging_steps=10,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        report_to="none",
    ),
)

# ✅ Focus training only on response generation
trainer = train_on_responses_only(
    trainer,
    instruction_part="<start_of_turn>user\n",
    response_part="<start_of_turn>model\n",
)

# ✅ Train the model
trainer_stats = trainer.train()

# ✅ Save LoRA adapters first (already in your script)
output_dir = "./gemma3-4b-it-finetuned"
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

# ✅ Merge LoRA adapters into the full model and save
merged_output_dir = "./gemma-3-finetune"  # Make sure this matches in both places
model.save_pretrained_merged(merged_output_dir, tokenizer)

# ✅ Convert the merged model to GGUF format using the same directory name
model.save_pretrained_gguf(
    merged_output_dir,  # This must be the same as the merged model path
    quantization_type="Q8_0"  # Options: "Q8_0", "BF16", "F16"
)

print(f"✅ Fine-tuning complete! Model saved in GGUF format at {merged_output_dir}")

