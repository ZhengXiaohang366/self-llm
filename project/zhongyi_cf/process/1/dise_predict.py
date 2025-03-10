from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import Trainer, TrainingArguments
import json, random
from datasets import Dataset
import os
from transformers import Qwen2ForSequenceClassification, Qwen2Tokenizer
import torch
import json

# 一些参数
model_path = '/dev/shm/zhengxiaohang/model/Qwen/Qwen2___5-0___5B-Instruct'
# model_name = "/qwen2.5-0.5-instruct/"  # 模型名或者本地路径
train_data_path = '../../dataset/prep/1/train.json'
test_file_path = '../../dataset/prep/1/test.json'
res_path = '../../dataset/res/tmp_res1.json'
save_path = "/dev/shm/zhengxiaohang/model/tmp_test/output/save_model1/"

num_train_epochs=10
per_device_train_batch_size=2
per_device_eval_batch_size=2
warmup_steps=50
weight_decay=0.01
logging_steps=1
use_cpu=False

# 创建标签到索引的映射
label_to_id = {
    "胸痹心痛病": 0,
    "心衰病": 1,
    "眩晕病": 2,
    "心悸病": 3
}

num_labels = len(label_to_id)  # 根据你的标签数量设置num_labels


# 加载模型和分词器
tokenizer = Qwen2Tokenizer.from_pretrained(save_path)
model = Qwen2ForSequenceClassification.from_pretrained(save_path)
for parameter in model.parameters():
    parameter.requires_grad = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 创建标签到索引的映射
label_to_id = {
    "胸痹心痛病": 0,
    "心衰病": 1,
    "眩晕病": 2,
    "心悸病": 3
}
# 创建 id_to_label 字典
id_to_label = {v: k for k, v in label_to_id.items()}

with open(test_file_path, 'r', encoding='utf-8') as file:
    # 使用 json.load() 方法将文件内容解析为 Python 对象
    data = json.load(file)

# 准备输入文本
texts = []
true_label = []
ID = []

for line in data:
    t = line['input']
    texts.append(t)
    # true_label.append(label_to_id[line['疾病']])
    ID.append(line['ID'])

# 对文本进行编码
inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
# 将输入数据移动到指定设备
inputs = {k: v.to(device) for k, v in inputs.items()}

# 进行推理
with torch.no_grad():
    outputs = model(**inputs)

# 获取预测结果
logits = outputs.logits
predictions = torch.argmax(logits, dim=-1)

results = []
for num, i in enumerate(predictions):
    results.append({"ID": ID[num], "疾病": id_to_label[i.item()]})

with open(res_path, 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=4)