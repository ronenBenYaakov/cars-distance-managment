from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import numpy as np


# Load LLM (flan-t5-small)
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
llm = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")



def generate_summary_llm(car_info):
    if not car_info:
        return "No vehicles detected."

    sorted_info = sorted(enumerate(car_info, start=1), key=lambda x: x[1][2])
    prompt_lines = []

    for i, (idx, (cx, cy, dist)) in enumerate(sorted_info):
        prompt_lines.append(f"Vehicle {idx} is at ({cx},{cy}) approximately {dist:.1f} meters away.")

    for i in range(len(sorted_info) - 1):
        idx1, (_, _, d1) = sorted_info[i]
        idx2, (_, _, d2) = sorted_info[i + 1]
        spacing = np.sqrt((d1 - d2)**2 + 4)
        prompt_lines.append(f"Vehicle {idx1} and {idx2} are {spacing:.1f} meters apart.")

    full_prompt = "Summarize this traffic scene:\n" + "\n".join(prompt_lines)
    inputs = tokenizer("Summarize: " + full_prompt, return_tensors="pt", truncation=True)
    outputs = llm.generate(**inputs, max_new_tokens=100)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)
