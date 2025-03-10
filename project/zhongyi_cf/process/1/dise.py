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

with open(train_data_path, 'r', encoding='utf-8') as file:
    # 使用 json.load() 方法将文件内容解析为 Python 对象
    data = json.load(file)

random.shuffle(data)



# 将文本标签转换为数值标签
for example in data:
    example['label'] = label_to_id[example['output']]

# 检查标签范围
for example in data:
    assert 0 <= example['label'] < len(label_to_id), f"Label out of range: {example['output']}"

train_data = []
for d in data:
    tmp = {}
    tmp['input'] = d['input']
    tmp['label'] = d['label']
    train_data.append(tmp)

# 将数据转换为datasets库的Dataset对象
dataset = Dataset.from_list(train_data)

# 将数据集拆分为训练集和验证集
dataset = dataset.train_test_split(test_size=0.1)

# 加载预训练的 Qwen2 模型和分词器
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=num_labels)
print(model)
model.config.pad_token_id = 151643
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 定义一个函数来处理数据集中的文本
def preprocess_function(examples):
    return tokenizer(examples['input'], truncation=True, padding=True, return_tensors="pt")

# 对数据集进行预处理
encoded_dataset = dataset.map(preprocess_function, batched=True)

# 定义训练参数
training_args = TrainingArguments(
    output_dir=save_path,                           # 输出目录
    num_train_epochs=num_train_epochs,              # 训练的epoch数
    per_device_train_batch_size=per_device_train_batch_size,    # 每个设备的训练batch size
    per_device_eval_batch_size=per_device_eval_batch_size,      # 每个设备的评估batch size
    warmup_steps=warmup_steps,                  # 预热步数
    weight_decay=weight_decay,                  # 权重衰减
    logging_dir=save_path,                      # 日志目录
    logging_steps=logging_steps,
    evaluation_strategy="epoch",
    save_strategy="epoch",    # 每个epoch保存一次检查点
    save_total_limit=3,       # 最多保存3个检查点，旧的会被删除
    use_cpu=False
)

# 定义Trainer
trainer = Trainer(
    model=model,                                    # 模型
    args=training_args,                             # 训练参数
    train_dataset=encoded_dataset['train'],         # 训练数据集
    eval_dataset=encoded_dataset['test']            # 评估数据集
)

# 打印训练集和验证集中的一些样本
print("Train dataset sample:")
print(encoded_dataset['train'][0])  # 打印训练集中的第一个样本

print("Eval dataset sample:")
print(encoded_dataset['test'][0])  # 打印验证集中的第一个样本

# 开始训练
trainer.train()
trainer.save_state()
trainer.save_model(output_dir=save_path)
tokenizer.save_pretrained(save_path)



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