from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from peft import PeftModel
import json
from tqdm import tqdm

version = '0308'

prep_test_file_path = '../dataset/prep/test_json.json'
res_file_path = f'../dataset/res/lora_res_test{version}.json'
res_txt_path = f'../dataset/res/lora_res_test{version}.txt'
model_path = '/dev/shm/zhengxiaohang/model/Qwen/Qwen2___5-7B-Instruct'
lora_path = "./output/Qwen2.5_instruct_lora"
use_path = lora_path+'/checkpoint-600'

if __name__=="__main__":

    with open(prep_test_file_path, 'r', encoding='utf-8') as file:
        test_data = json.load(file)
    print(test_data[0])


    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto",torch_dtype=torch.bfloat16, trust_remote_code=True).eval()
    model = PeftModel.from_pretrained(model, model_id=use_path)



    for d in tqdm(test_data, desc="Processing test data"):
        prompt = d['input']
        inputs = tokenizer.apply_chat_template(
            [{"role": "user", "content": d['instruction']}, {"role": "user", "content": prompt}],
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
            return_dict=True
            ).to('cuda')

        gen_kwargs = {"max_length": 2500, "do_sample": True, "top_k": 1}
        with torch.no_grad():
            outputs = model.generate(**inputs, **gen_kwargs)
            outputs = outputs[:, inputs['input_ids'].shape[1]:]
            d['output'] = tokenizer.decode(outputs[0], skip_special_tokens=True)

    with open(res_file_path, 'w', encoding='utf-8') as file:
        json.dump(test_data, file, ensure_ascii=False, indent=4)

    # 打开文件以写入模式
    with open(res_txt_path, 'w', encoding='utf-8') as file:
        # 遍历列表中的每个元素
        for line in test_data:
            # 将元素写入文件，并添加换行符
            file.write(line['output'] + '\n')
