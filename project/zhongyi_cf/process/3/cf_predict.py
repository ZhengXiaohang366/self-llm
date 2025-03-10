from datasets import Dataset
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForSeq2Seq, TrainingArguments, Trainer, GenerationConfig
import torch
from peft import LoraConfig, TaskType, get_peft_model


model_path = '/dev/shm/zhengxiaohang/model/Qwen/Qwen2___5-7B-Instruct'

train_data_path = '../../dataset/prep/3/train.json'
test_file_path = '../../dataset/prep/3/test.json'
res_path = '../../dataset/res/tmp_res3.json'
save_path = "/dev/shm/zhengxiaohang/model/tmp_test/output/save_model3/"

lora_path = save_path+'checkpoint-1000'

config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    inference_mode=False, # 训练模式
    r=8, # Lora 秩
    lora_alpha=32, # Lora alaph，具体作用参见 Lora 原理
    lora_dropout=0.1# Dropout 比例
)
args = TrainingArguments(
    output_dir=save_path,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=2,
    logging_steps=1,
    num_train_epochs=5,
    save_steps=10,
    learning_rate=5e-5,
    save_on_each_node=True,
    gradient_checkpointing=True
)

import json

with open(test_file_path, 'r', encoding='utf-8') as file:
    # 读取并解析 JSON 数据
    test_data = json.load(file)
    # 打印读取的数据
    print(test_data[0])

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from peft import PeftModel

# 加载tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

# 加载模型
model = AutoModelForCausalLM.from_pretrained(model_path,  device_map="cuda:0",torch_dtype=torch.bfloat16, trust_remote_code=True).eval()

# 加载lora权重
model = PeftModel.from_pretrained(model, model_id=lora_path)

from tqdm import tqdm

# 假设 test_data、tokenizer 和 model 已经定义
for d in tqdm(test_data, desc="Processing test data"):
    prompt = d['input']
    inputs = tokenizer.apply_chat_template(
        [{"role": "user", "content": d['instruction']}, {"role": "user", "content": prompt}],
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
        return_dict=True
        ).to('cuda')

    gen_kwargs = {"max_length": 4096, "do_sample": True, "top_k": 1}
    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)
        outputs = outputs[:, inputs['input_ids'].shape[1]:]
        d['output'] = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # 方法一：使用 with 语句自动管理文件的打开和关闭


with open(res_path, 'w', encoding='utf-8') as file:
    # 将 list 转换为 JSON 格式并写入文件
    json.dump(test_data, file, ensure_ascii=False, indent=4)


