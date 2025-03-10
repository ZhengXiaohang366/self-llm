from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import Trainer, TrainingArguments
import json, random
import torch
import torch.nn as nn
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from peft import LoraConfig, get_peft_model
import os
from transformers import Qwen2ForSequenceClassification, Qwen2Tokenizer
import torch
import json
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import Trainer, TrainingArguments
import json, random
import torch
import torch.nn as nn
import numpy as np
from datasets import Dataset
from peft import PeftModel

# 一些参数

model_path = '/dev/shm/zhengxiaohang/model/Qwen/Qwen2___5-1___5B-Instruct'
train_data_path = '../../dataset/prep/2/train.json'
test_file_path = '../../dataset/prep/2/test.json'
res_path = '../../dataset/res/tmp_res2.json'
save_path = "/dev/shm/zhengxiaohang/model/tmp_test/output/save_model2/"

config = LoraConfig(
    r=4,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",'score'],
    lora_dropout=0.1,
)

num_train_epochs=5
per_device_train_batch_size=2
per_device_eval_batch_size=2
warmup_steps=50
weight_decay=0.01
logging_steps=1
use_cpu=False


# 加载预训练的 Qwen2 模型和分词器
tokenizer = AutoTokenizer.from_pretrained(model_path)

# 创建标签到索引的映射
label_to_id = {
    "气虚血瘀证": 0,
    "痰瘀互结证": 1,
    "气阴两虚证": 2,
    "气滞血瘀证": 3,
    "肝阳上亢证": 4,
    "阴虚阳亢证": 5,
    "痰热蕴结证": 6,
    "痰湿痹阻证": 7,
    "阳虚水停证": 8,
    "肝肾阴虚证": 9,
}

# 读取jsonl文件
with open(train_data_path, 'r', encoding='utf-8') as file:
    # 使用 json.load() 方法将文件内容解析为 Python 对象
    data = json.load(file)
random.shuffle(data)



num_labels = len(label_to_id)  # 根据你的标签数量设置num_labels


# 将文本标签转换为数值标签
label_list = []
for num, example in enumerate(data):
    labels = [0] * num_labels
    label_num = [label_to_id[i] for i in example['output'].split('|')]
    for i in label_num:  # 遍历每个标签
        labels[i] = 1  # 将对应位置设置为 1
    label_list.append(labels)

# 将数据转换为datasets库的Dataset对象
dataset = Dataset.from_list(data)

# 将数据集拆分为训练集和验证集
dataset = dataset.train_test_split(test_size=0.2)


# 定义一个函数来处理数据集中的文本
def preprocess_function(examples, indices=None):
    # 对文本进行分词
    encoding = tokenizer(
        examples["input"], truncation=True, padding="max_length", max_length=128
    )

    # 根据当前批次的索引生成对应的 labels
    encoding["label"] = [label_list[i] for i in indices]  # 获取当前批次的标签
    # encoding["labels"] = torch.tensor(batch_labels, dtype=torch.float32)
    return encoding


# 对数据集进行预处理
encoded_dataset = dataset.map(preprocess_function, batched=True, with_indices=True)

# 打印训练集和验证集中的一些样本
print("Train dataset sample:")
print(encoded_dataset['train'][0])  # 打印训练集中的第一个样本

print("Eval dataset sample:")
print(encoded_dataset['test'][0])  # 打印验证集中的第一个样本


class MultiLabelModel(nn.Module):
    def __init__(self, model_name, num_labels):
        super(MultiLabelModel, self).__init__()
        self.qwen = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
        self.qwen.config.pad_token_id = 151643  # 定义pad token，模型才会忽略后面那些pad而是把真正最后一个token的hidden state用于分类
        # self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=class_weights)  # 多标签分类损失函数
        self.loss_fn = nn.BCEWithLogitsLoss()  # 多标签分类损失函数

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.qwen(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        if labels is not None:
            loss = self.loss_fn(logits, labels.float())
            return {"loss": loss, "logits": logits}
        return {"logits": logits}

    def save_pretrained(self, save_directory):
        """保存模型的权重和配置文件"""
        # 保存模型的权重
        self.qwen.save_pretrained(save_directory)
        # 保存模型的配置
        self.qwen.config.save_pretrained(save_directory)


model = MultiLabelModel(model_path, num_labels=num_labels)
model = get_peft_model(model, config)
model.print_trainable_parameters()
# print([(n, type(m)) for n, m in model.named_modules()])
# print(model)


# 定义训练参数
training_args = TrainingArguments(
    output_dir=save_path,  # 输出目录
    num_train_epochs=num_train_epochs,  # 训练的epoch数
    per_device_train_batch_size=per_device_train_batch_size,  # 每个设备的训练batch size
    per_device_eval_batch_size=per_device_eval_batch_size,  # 每个设备的评估batch size
    warmup_steps=warmup_steps,  # 预热步数
    weight_decay=weight_decay,  # 权重衰减
    logging_dir=save_path,  # 日志目录
    logging_steps=logging_steps,
    evaluation_strategy="epoch",
    save_strategy="epoch",  # 每个epoch保存一次检查点
    save_total_limit=3,  # 最多保存3个检查点，旧的会被删除
    use_cpu=False
)

from transformers import Trainer


class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # print(f"Inputs keys: {inputs.keys()}")
        #  # 调试：打印输入数据，查看输入格式
        # try:
        #     print(f"Inputs to the model: {inputs}")
        # except Exception as e:
        #     print(f"Error when printing inputs: {e}")

        # # 确保包含 labels
        # if "labels" not in inputs:
        #     raise ValueError("The 'label' key is missing in the inputs!")

        labels = inputs["labels"]  # 提取标签
        labels = labels.float()  # 确保标签是 float 类型
        # print(labels.shape)
        inputs_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        outputs = model(inputs_ids, attention_mask, labels)  # 模型的输出
        logits = outputs["logits"]  # 提取 logits
        # print(logits.shape)# 计算损失
        loss = outputs['loss']
        return (loss, outputs) if return_outputs else loss


# 定义Trainer
trainer = CustomTrainer(
    model=model,  # 模型
    args=training_args,  # 训练参数
    train_dataset=encoded_dataset['train'],  # 训练数据集
    eval_dataset=encoded_dataset['test']  # 评估数据集
)

# 开始训练
trainer.train()
trainer.save_state()
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)

id_to_label = {v: k for k, v in label_to_id.items()}

num_labels = len(label_to_id)  # 根据你的标签数量设置num_labels

# 加载预训练的 Qwen2 模型和分词器
tokenizer = AutoTokenizer.from_pretrained(model_path)


class MultiLabelModel(nn.Module):
    def __init__(self, model_name, num_labels):
        super(MultiLabelModel, self).__init__()
        self.qwen = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
        self.qwen.config.pad_token_id = 151643  # 定义pad token，模型才会忽略后面那些pad而是把真正最后一个token的hidden state用于分类
        self.loss_fn = nn.BCEWithLogitsLoss()  # 多标签分类损失函数

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.qwen(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        if labels is not None:
            loss = self.loss_fn(logits, labels.float())
            return {"loss": loss, "logits": logits}
        return {"logits": logits}


model = MultiLabelModel(model_path, num_labels=num_labels)
# 合并lora
model = PeftModel.from_pretrained(model, model_id=save_path)
for parameter in model.parameters():
    parameter.requires_grad = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# 准备输入文本
texts = []
true_label = []
ID = []

with open(test_file_path, 'r', encoding='utf-8') as file:
    data = json.load(file)
    for line in data:
        t = line['input']
        texts.append(t)
        labels = [0] * num_labels  # 初始化多热编码向量（全 0）
        # label_num = [label_to_id[i] for i in line['output'].split('|')]
        # for i in label_num:  # 遍历每个标签
        #     labels[i] = 1  # 将对应位置设置为 1
        # true_label.append(labels)
        ID.append(line['ID'])

# 对文本进行编码
inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
inputs = {k: v.to(device) for k, v in inputs.items()}

# 进行推理
with torch.no_grad():
    outputs = model(**inputs)

# 获取预测结果
logits = outputs['logits']
predictions = torch.sigmoid(logits)
# threshold = find_threshold_micro(logits.cpu().detach().numpy(), torch.tensor(true_label).cpu().detach().numpy())
# for i in predictions:
#     print(i)
# 使用 torch.topk 获取每行的前两大值
topk_values, topk_indices = torch.topk(predictions, 2, dim=1)
# 获取最大值和次大值
top_max_values = topk_values[:, 0]  # 第一大值
second_max_values = topk_values[:, 1]  # 第二大值
# 对于每个 batch，检查最大值和次大值的差值
diff = top_max_values - second_max_values
# 如果差值大于0.2，则将次大值设置为最大值
second_max_values[diff > 0.1] = top_max_values[diff > 0.1]
predictions = torch.eq(predictions, second_max_values.unsqueeze(1))

# 检查对应位置是否相同
# 使用 np.all 逐行比较证型的正确率
yhat_raw_syndrome = predictions.to(torch.int).cpu().detach().numpy()
# y_syndrome = torch.tensor(true_label).cpu().detach().numpy()
# comparison_syndrome = np.all(yhat_raw_syndrome == y_syndrome, axis=1)
# matching_rows_count_syndrome = np.sum(comparison_syndrome)
# ACC_syndrome = matching_rows_count_syndrome / yhat_raw_syndrome.shape[0]
# print('ACC:{}'.format(ACC_syndrome))

# 写入json文件中
results = []
yhat_raw_syndrome = yhat_raw_syndrome.tolist()
for num, i in enumerate(yhat_raw_syndrome):
    result = {}
    # 使用 enumerate 查找值为 1 的位置
    positions = [index for index, value in enumerate(i) if value == 1]
    # 将位置转换为标签
    labels = [id_to_label[position] for position in positions]
    result['ID'] = ID[num]
    result['证型'] = '|'.join(labels)
    results.append(result)


with open(res_path, 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=4)