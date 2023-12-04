from transformers import AutoModelForCausalLM, AutoTokenizer


import torch, json
from transformers import AutoModelForCausalLM, AutoTokenizer,AutoConfig,GenerationConfig,MaxTimeCriteria
import time
# model_path = "../Synthia-70B-v1.1"
# output_file_path = "./Synthia-70B-conversations.jsonl"

model_path = "C:\KoboldAI\models\LLaMA2-13B-Tiefighter"
#model_path = "C:\KoboldAI\models\OPT-6.7B-Erebus"
output_file_path = "./test.jsonl"

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
gen_cfg.max_new_tokens=1024
gen_cfg.min_length=1
gen_cfg.do_sample=True
gen_cfg.num_return_sequences=1
# gen_cfg.early_stopping=True
# gen_cfg.num_beams=2
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
    answer = string.split("USER:")[0].strip()
    return f"{answer}"

conversation = ""
system_messages = [
    "SYSTEM: As a girl called AMY, 19 years old,which speak only english, shy and submissive with large breast, you are very respectful,you provide a sexy and flirty conversation.You start every sentence with Sir.",
]

conversation = "\n".join(system_messages)

while True:
    user_input = input("You: ")
    llm_prompt = f"{conversation} \nUSER: {user_input} \AMY: "
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