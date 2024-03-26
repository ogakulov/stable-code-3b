import argparse
import os
import spaces


import gradio as gr

import json
from threading import Thread
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

MAX_LENGTH = 4096
DEFAULT_MAX_NEW_TOKENS = 1024


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str)  # model path
    parser.add_argument("--n_gpus", type=int, default=1)  # n_gpu
    return parser.parse_args()

@spaces.GPU()
def predict(message, history, system_prompt, temperature, max_tokens):
    global model, tokenizer, device
    instruction = "<|im_start|>system\nA chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.\n<|im_end|>\n"
    for human, assistant in history:
        instruction += '<|im_start|>user\n' + human + '\n<|im_end|>\n<|im_start|>assistant\n' + assistant
    instruction += '\n<|im_start|>user\n' + message + '\n<|im_end|>\n<|im_start|>assistant\n'
    problem = [instruction]
    stop_tokens = ["<|endoftext|>", "<|im_end|>"]
    streamer = TextIteratorStreamer(tokenizer, timeout=100.0, skip_prompt=True, skip_special_tokens=True)
    enc = tokenizer(problem, return_tensors="pt", padding=True, truncation=True)
    input_ids = enc.input_ids
    attention_mask = enc.attention_mask

    if input_ids.shape[1] > MAX_LENGTH:
        input_ids = input_ids[:, -MAX_LENGTH:]

    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    generate_kwargs = dict(
        {"input_ids": input_ids, "attention_mask": attention_mask},
        streamer=streamer,
        do_sample=True,
        top_p=0.95,
        temperature=0.5,
        max_new_tokens=DEFAULT_MAX_NEW_TOKENS,
    )
    t = Thread(target=model.generate, kwargs=generate_kwargs)
    t.start()
    outputs = []
    for text in streamer:
        outputs.append(text)
        if text in stop_tokens:
            break
        print(text)
        yield "".join(outputs)



if __name__ == "__main__":
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained("stabilityai/stable-code-instruct-3b")
    model = AutoModelForCausalLM.from_pretrained("stabilityai/stable-code-instruct-3b", torch_dtype=torch.bfloat16)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    gr.ChatInterface(
        predict,
        title="Stable Code Instruct Chat - Demo",
        description="Chat Model Stable Code 3B",
        theme="soft",
        chatbot=gr.Chatbot(label="Chat History",),
        textbox=gr.Textbox(placeholder="input", container=False, scale=7),
        retry_btn=None,
        undo_btn="Delete Previous",
        clear_btn="Clear",
        additional_inputs=[
            gr.Textbox("A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.", label="System Prompt"),
            gr.Slider(0, 1, 0.9, label="Temperature"),
            gr.Slider(100, 2048, 1024, label="Max Tokens"),
        ],
        additional_inputs_accordion_name="Parameters",
    ).queue().launch()
