from transformers import AutoModelForCausalLM, AutoTokenizer


import torch, json
from transformers import AutoModelForCausalLM, AutoTokenizer,GenerationConfig
import time
import os
# model_path = "../Synthia-70B-v1.1"
# output_file_path = "./Synthia-70B-conversations.jsonl"

#model_path = "C:\KoboldAI\models\openchat_3.5"
#model_path = "C:\KoboldAI\models\OPT-6.7B-Erebus"

#get the parameter -m from the command line to the absolute path of the model
#also get the parameter -n without any value to start a new conversation
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", help="absolute path to the model")
parser.add_argument("-s", "--startprompt", help="absolute path to start prompt")

parser.add_argument("-n", "--new", help="start a new conversation", action="store_true")
args = parser.parse_args()
model_path = args.model
new_conversation = args.new
start_prompt_path = args.startprompt
if model_path is None:
    print("Please specify the absolute path to the model with -m")
    quit()
if start_prompt_path is None:
    print("Please specify the absolute path to the start prompt with -s")
    quit()
if not os.path.exists(model_path):
    print("The model path does not exist")
    quit()
if not os.path.exists(start_prompt_path):
    print("The start prompt path does not exist")
    quit()

output_file_path = "./test2.jsonl"
if new_conversation:
    #delete the file if it exists
    if os.path.exists(output_file_path):
        os.remove(output_file_path)

    

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
    # offload_folder="offload"
)

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
gen_cfg = GenerationConfig.from_model_config(model.config)
# model.config.max_new_tokens = 30
# model.config.min_length = 1
gen_cfg.max_new_tokens=512
gen_cfg.min_length=1
gen_cfg.do_sample=True
gen_cfg.num_return_sequences=1
# gen_cfg.early_stopping=True
# gen_cfg.num_beams=2
gen_cfg.max_time=60
print(model.generation_config)
def generate_text(instruction):
    
    input_ids = tokenizer.encode(instruction, return_tensors="pt")
    input_ids = input_ids.to("cuda:0")  # move to GPU
    #max_time = MaxTimeCriteria(max_time=20)
    
    out = model.generate(
        input_ids,
         generation_config=gen_cfg
         
    )
    length = len(input_ids[0])
    output = out[0][length:]
    string = tokenizer.decode(output, skip_special_tokens=True)
    print("full answer: ")
    print(string)
    answer = string.split("USER:")[0].strip()
    
    return f"{answer}"

conversation = ""
with open (start_prompt_path, "r",encoding='utf-8') as myfile:
    system_messages = myfile.readlines()
conversation = "\n".join(system_messages)
print("system messages:")
print(conversation)
while True:
    user_input = input("You: ")
    llm_prompt = f"{conversation} \nUSER: {user_input} \nASSISTANT: "
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
    splitted = answer.split("ASSISTANT:")
    for sentences in splitted:
        print("ASSISTANT: " + sentences)
    conversation = f"{llm_prompt}{answer}"
    json_data = {"prompt": user_input, "answer": answer}

    ## Save your conversation
    with open(output_file_path, "a") as output_file:
        output_file.write(json.dumps(json_data) + "\n")