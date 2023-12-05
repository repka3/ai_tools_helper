from transformers import AutoModelForCausalLM, AutoTokenizer


import json
from transformers import AutoModelForCausalLM, AutoTokenizer,GenerationConfig
import time
import os

import argparse



output_file_path = "./test_wizard.jsonl"
model_name_or_path="C:\KoboldAI\models\WizardLM-7B-uncensored-GPTQ"

model = AutoModelForCausalLM.from_pretrained(model_name_or_path,
                                             device_map="auto",
                                             trust_remote_code=False,
                                             revision="oobaCUDA")

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)

# gen_cfg = GenerationConfig.from_model_config(model.config)
# # model.config.max_new_tokens = 30
# # model.config.min_length = 1
# gen_cfg.max_new_tokens=1024
# gen_cfg.min_length=1
# gen_cfg.do_sample=True
# gen_cfg.num_return_sequences=1
# # gen_cfg.early_stopping=True
# # gen_cfg.num_beams=2
# gen_cfg.max_time=30
print(model.generation_config)
def generate_text(instruction):
    
    input_ids = tokenizer.encode(instruction, return_tensors="pt")
    input_ids = input_ids.to("cuda:0")  # move to GPU
    #max_time = MaxTimeCriteria(max_time=20)
    print(instruction)
    out = model.generate(inputs=input_ids, temperature=0.7, do_sample=True, top_p=0.95, top_k=40, max_new_tokens=256, min_length=1, num_return_sequences=1, max_time=60)

    length = len(input_ids[0])
    output = out[0][length:]
    string = tokenizer.decode(output, skip_special_tokens=True)
    print(string)
    answer = string.split("USER:")[0].strip()
    return f"{answer}"

conversation = ""
with open ("start_inputs_test2.txt", "r") as myfile:
    system_messages = myfile.readlines()
conversation = "\n".join(system_messages)
# print("system messages:")
# print(conversation)
while True:
    user_input = input("You: ")
    llm_prompt = f"{conversation} \nUSER: {user_input} \n"
    # calculate elapsed time
    start=time.time()
    answer = generate_text(llm_prompt)
    stop=time.time()
    elapsed=stop-start
    #converte elapsed to HH:MM:SS
    m, s = divmod(elapsed, 60)
    h, m = divmod(m, 60)
    elapsed_time = "%d:%02d:%02d" % (h, m, s)
    print("Elapsed time: " + elapsed_time)
    print(answer)
    conversation = f"{llm_prompt}{answer}"
    json_data = {"prompt": user_input, "answer": answer}

    ## Save your conversation
    with open(output_file_path, "a") as output_file:
        output_file.write(json.dumps(json_data) + "\n")