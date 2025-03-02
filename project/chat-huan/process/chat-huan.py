from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import streamlit as st
from peft import PeftModel

# 在侧边栏中创建一个标题和一个链接
with st.sidebar:
    st.markdown("## chat-huan LLM")
    # 创建一个滑块，用于选择最大长度，范围在 0 到 8192 之间，默认值为 512（Qwen2.5 支持 128K 上下文，并能生成最多 8K tokens）
    max_length = st.slider("max_length", 0, 8192, 512, step=1)

# 创建一个标题和一个副标题
st.title("💬 甄嬛")
st.caption("🚀 A streamlit chatbot powered by huanhuan")

# 定义一个函数，用于获取模型和 tokenizer
@st.cache_resource
def get_model():
    mode_path = '/root/autodl-tmp/self-llm/model/Qwen/Qwen2.5-7B-Instruct'
    lora_path = './output/Qwen2.5_instruct_lora/checkpoint-600'  # 这里改称你的 lora 输出对应 checkpoint 地址

    # 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(mode_path, trust_remote_code=True)

    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(mode_path, torch_dtype=torch.bfloat16, trust_remote_code=True)

    # 将模型移动到 GPU
    model = model.to('cuda')

    # 加载 lora 权重
    model = PeftModel.from_pretrained(model, model_id=lora_path)

    # 确保 Lora 模型也在 GPU 上
    model = model.to('cuda')

    return tokenizer, model

# 加载 Qwen2.5 的 model 和 tokenizer
tokenizer, model = get_model()

# 如果 session_state 中没有 "messages"，则创建一个包含默认消息的列表
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "贵人安好，贵人可有吩咐？臣妾必定尽力而为 "}]

# 遍历 session_state 中的所有消息，并显示在聊天界面上
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# 如果用户在聊天输入框中输入了内容，则执行以下操作
if prompt := st.chat_input():
    # 在聊天界面上显示用户的输入
    st.chat_message("user").write(prompt)

    # 将用户输入添加到 session_state 中的 messages 列表中
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 手动添加“假设你是皇帝身边的女人--甄嬛。”提示信息
    messages_with_prompt = [{"role": "user", "content": "假设你是皇帝身边的女人--甄嬛。"}] + st.session_state.messages

    # 将对话输入模型，获得返回
    input_ids = tokenizer.apply_chat_template(messages_with_prompt, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([input_ids], return_tensors="pt").to('cuda')
    generated_ids = model.generate(model_inputs.input_ids, max_new_tokens=max_length)
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    # 将模型的输出添加到 session_state 中的 messages 列表中
    st.session_state.messages.append({"role": "assistant", "content": response})
    # 在聊天界面上显示模型的输出
    st.chat_message("assistant").write(response)
    # print(st.session_state) # 打印 session_state 调试