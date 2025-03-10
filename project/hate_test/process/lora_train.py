from datasets import Dataset
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForSeq2Seq, TrainingArguments, Trainer, GenerationConfig
import torch
from peft import LoraConfig, TaskType, get_peft_model
import json

raw_train_file_path = '../dataset/raw/train.json'
raw_test_file_path = '../dataset/raw/test1.json'
prep_train_file_path = '../dataset/prep/train_json.json'
prep_test_file_path = '../dataset/prep/test_json.json'
model_path = '/dev/shm/zhengxiaohang/model/Qwen/Qwen2___5-7B-Instruct'

config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    inference_mode=False,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1
)

args = TrainingArguments(
    output_dir="./output/Qwen2.5_instruct_lora",
    per_device_train_batch_size=8,
    gradient_accumulation_steps=4,
    logging_steps=10,
    num_train_epochs=5,
    save_steps=100,
    learning_rate=1e-4,
    save_on_each_node=True,
    gradient_checkpointing=True
)

def process_func(example):
    MAX_LENGTH = 4096
    input_ids, attention_mask, labels = [], [], []
    instruction = tokenizer(
        f"<|im_start|>system\n现在你要进行一个细粒度中文仇恨识别任务<|im_end|>\n<|im_start|>user\n{example['instruction'] + example['input']}<|im_end|>\n<|im_start|>assistant\n",
        add_special_tokens=False) 
    response = tokenizer(f"{example['output']}", add_special_tokens=False)
    input_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id]
    attention_mask = instruction["attention_mask"] + response["attention_mask"] + [1]
    labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [tokenizer.pad_token_id]
    if len(input_ids) > MAX_LENGTH:  
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }

if __name__=="__main__":

    print('数据预览')
    with open(raw_train_file_path, 'r', encoding='utf-8') as file:
        train_data = json.load(file)
        print(train_data[0])

    with open(raw_test_file_path, 'r', encoding='utf-8') as file:
        test_data = json.load(file)
        print(test_data[0])

    hate_set = []
    for d in train_data:
        hate_set.append(d['output'].split('|')[2].split(',')[0])
    target_group_list = ['LGBTQ','Racism','Region','Sexism','others','non-hate']


    train_json = []
    for d in train_data:
        tmp_dic = {}
        tmp_dic["instruction"] = '''请从中文社交媒体文本中识别仇恨言论四元组。按以下要求处理：\n1.识别评论对象(Target)，无目标时写NULL\n2.提取对应论点(Argument)\n3.确定目标群体(Targeted Group)：LGBTQ/Racism/Region/Sexism/others\n4.判断是否仇恨(Hateful)：hate/Non-hate\n\n输出格式：Target | Argument | Targeted Group | Hateful [SEP]... [END]\n多个四元组用[SEP]分隔，最后用[END]结尾'''
        tmp_dic["input"] = d['content']
        tmp_dic["output"] = d['output']
        train_json.append(tmp_dic)

    test_json = []
    for d in test_data:
        tmp_dic = {}
        tmp_dic["instruction"] = '''请从中文社交媒体文本中识别仇恨言论四元组。按以下要求处理：\n1.识别评论对象(Target)，无目标时写NULL\n2.提取对应论点(Argument)\n3.确定目标群体(Targeted Group)：LGBTQ/Racism/Region/Sexism/others\n4.判断是否仇恨(Hateful)：hate/Non-hate\n\n输出格式：Target | Argument | Targeted Group | Hateful [SEP]... [END]\n多个四元组用[SEP]分隔，最后用[END]结尾'''
        tmp_dic['id'] = d['id']
        tmp_dic["input"] = d['content']
        tmp_dic["output"] = ""
        test_json.append(tmp_dic)

    with open(prep_train_file_path, 'w', encoding='utf-8') as file:
        json.dump(train_json, file, ensure_ascii=False, indent=4)

    with open(prep_test_file_path, 'w', encoding='utf-8') as file:
        json.dump(test_json, file, ensure_ascii=False, indent=4)

    df = pd.read_json(prep_train_file_path)
    ds = Dataset.from_pandas(df)
    print('token转化中')
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, trust_remote_code=True)
    tokenized_id = ds.map(process_func, remove_columns=ds.column_names)
    print('token结果校验')
    print(tokenizer.decode(tokenized_id[0]['input_ids']))
    print(tokenizer.decode(list(filter(lambda x: x != -100, tokenized_id[1]["labels"]))))

    print('模型载入')
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto",torch_dtype=torch.bfloat16)
    model.enable_input_require_grads()
    model = get_peft_model(model, config)

    model.print_trainable_parameters()
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_id,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
    )

    trainer.train()