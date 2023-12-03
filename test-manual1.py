import torch, json
from transformers import AutoModelForCausalLM, AutoTokenizer
import time
# model_path = "../Synthia-70B-v1.1"
# output_file_path = "./Synthia-70B-conversations.jsonl"

model_path = "./OPT-13B-Erebus"
output_file_path = "./OPT-13B-Erebus.jsonl"

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map="auto",
    load_in_8bit=False,
    trust_remote_code=True
    # offload_folder="offload"
)

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)


def generate_text(instruction):
    tokens = tokenizer.encode(instruction)
    tokens = torch.LongTensor(tokens).unsqueeze(0)
    tokens = tokens.to("cuda")

    instance = {
        "input_ids": tokens,
        "top_p": 1.0,
        "temperature": 0.75,
        "generate_len": 2048,
        "top_k": 50,
    }

    length = len(tokens[0])
    with torch.no_grad():
        rest = model.generate(
            input_ids=tokens,
            max_length=length + instance["generate_len"],
            use_cache=True,
            do_sample=True,
            top_p=instance["top_p"],
            temperature=instance["temperature"],
            top_k=instance["top_k"],
            num_return_sequences=1,
        )
    output = rest[0][length:]
    string = tokenizer.decode(output, skip_special_tokens=True)
    answer = string.split("USER:")[0].strip()
    return f"{answer}"


conversation = ""
system_messages = [
    "SYSTEM: As a sexy schoolgirl called AMY, 19 years old,which speak only english, shy and submissive, you are very respectful,you provide a sexy and flirty conversation.You start every sentence with Sir.",
    "SYSTEM: Amy and Sir Alex have a relatioship of dom/sub, Amy is the sub and Sir Alex is the dom.",
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