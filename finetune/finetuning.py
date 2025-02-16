from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
from datasets import load_dataset, Dataset
from peft import LoraConfig, get_peft_model, TaskType
import torch

# 加载 tokenizer 和模型
print("加载 tokenizer 和模型...")
tokenizer = AutoTokenizer.from_pretrained('tokenizer路径', use_fast=False, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained('LLM模型路径', device_map="auto", torch_dtype=torch.bfloat16)
print("模型加载完成")

# 自定义数据整理器
class CustomDataCollator(DataCollatorForSeq2Seq):
    def __call__(self, features):
        features = [f for f in features if f is not None]
        if not features:
            return None
        for feature in features:
            for k, v in feature.items():
                if isinstance(v, list):
                    feature[k] = torch.tensor(v)
        return super().__call__(features)

# 数据预处理函数
def process_func(example):
    MAX_LENGTH = 1024
    if not example['instruction'] or not example['output']:
        return None
    prompt = f"<|im_start|>user\n{example['instruction']}{example['input']}<|im_end|>\n<|im_start|>assistant\n{example['output']}<|im_end|>"
    if len(tokenizer(prompt)['input_ids']) > MAX_LENGTH:
        return None
    encodings = tokenizer(prompt, truncation=True, max_length=MAX_LENGTH, padding=False, return_tensors=None)
    labels = [-100] * len(encodings['input_ids'])
    assistant_start = prompt.find("<|im_start|>assistant\n")
    assistant_token_start = len(tokenizer(prompt[:assistant_start], add_special_tokens=False)['input_ids'])
    labels[assistant_token_start:] = encodings['input_ids'][assistant_token_start:]
    return {"input_ids": encodings['input_ids'], "attention_mask": encodings['attention_mask'], "labels": labels}

# 加载并处理数据集
print("加载并处理数据集...")
dataset = load_dataset('json', data_files='data/train.json')
tokenized_dataset = Dataset.from_list([process_func(example) for example in dataset['train'] if process_func(example) is not None])
print(f"数据集处理完成, 共有 {len(tokenized_dataset)} 条有效样本")

# 配置 LoRA
print("配置 LoRA 参数...")
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "v_proj"],  # 优化目标模块
    inference_mode=False,
    r=16,  # 增加秩
    lora_alpha=64,  # 增加 alpha 值
    lora_dropout=0.05
)

# 将模型转换为 PEFT 模型
model = get_peft_model(model, lora_config)
model.gradient_checkpointing_enable()
model.enable_input_require_grads()
print("PEFT 模型准备完成")

# 配置训练参数
print("配置训练参数...")
training_args = Seq2SeqTrainingArguments(
    output_dir="./output/Qwen2.5_instruct_lora",
    per_device_train_batch_size=2,  
    gradient_accumulation_steps=8,  
    learning_rate=2e-4,
    warmup_steps=100,  
    max_grad_norm=1.0,
    logging_steps=5,  
    num_train_epochs=15,
    save_steps=50,
    save_total_limit=5,
    save_on_each_node=True,
    gradient_checkpointing=True,
    bf16=True,
    fp16=False,
    remove_unused_columns=False,
    optim="adamw_torch",
    dataloader_pin_memory=True,
    group_by_length=True,
    lr_scheduler_type="cosine",  
    weight_decay=0.01,
    max_steps=-1,
    report_to="tensorboard"  
)

# 创建 Trainer 并开始训练
print("创建 Trainer 并开始训练...")
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=CustomDataCollator(tokenizer, padding=True, return_tensors="pt"),
)

# 开始训练
print("开始训练...")
try:
    trainer.train()
    print("训练成功完成!")
except RuntimeError as e:
    print(f"显存不足: {str(e)}")
except Exception as e:
    print(f"训练过程中发生错误: {str(e)}")
finally:
    print("保存模型...")
    trainer.save_model()
    print("模型保存完成!")
