import json

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

device = "cpu"

tokenizer = AutoTokenizer.from_pretrained("humarin/chatgpt_paraphraser_on_T5_base")

model = AutoModelForSeq2SeqLM.from_pretrained("humarin/chatgpt_paraphraser_on_T5_base").to(device)


def paraphrase(
        question,
        num_beams=5,
        num_beam_groups=5,
        num_return_sequences=2,
        repetition_penalty=10.0,
        diversity_penalty=3.0,
        no_repeat_ngram_size=2,
        temperature=0.7,
        max_length=128
):
    input_ids = tokenizer(
        f'paraphrase: {question}',
        return_tensors="pt", padding="longest",
        max_length=max_length,
        truncation=True,
    ).input_ids

    outputs = model.generate(
        input_ids, temperature=temperature, repetition_penalty=repetition_penalty,
        num_return_sequences=num_return_sequences, no_repeat_ngram_size=no_repeat_ngram_size,
        num_beams=num_beams, num_beam_groups=num_beam_groups,
        max_length=max_length, diversity_penalty=diversity_penalty
    )

    res = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    return res[1]


def process_json(input_file, output_file):
    new_data = []
    with open(input_file, 'r') as f:
        data = json.load(f)

    for index, entry in enumerate(data):
        question = entry.get("question")
        best_answer = entry.get("human_answer")

        # Call your function to modify the answer
        generated_answer = paraphrase(best_answer)

        # Add the generated_answer to the entry
        entry["generated_answer"] = generated_answer
        new_entry = {"question" : question, "human_answer" : best_answer , "generated_answer" : generated_answer}
        new_data.append(new_entry)
    with open(output_file, 'w') as f:
        json.dump(new_data, f, indent=2)


input_json_file = "t1.json"
output_json = "dataset.json"

#process_json(input_json_file, output_json)

def process_json1(input_file, output_file):
    new_data = []
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    for index, entry in enumerate(data):
        if index > 1000:
            break
        question = entry.get("question")
        human_answers = entry.get("human_answers")
        chatgpt_answers = entry.get("chatgpt_answers")

        new_entry = {"question" : question, "human_answer" : human_answers , "generated_answer" : chatgpt_answers}
        new_data.append(new_entry)
    with open(output_file, 'w',encoding='utf-8') as f:
        json.dump(new_data, f, indent=2)


def decode(str):
    return str.encode('utf-8').decode('unicode_escape')


def remove_empty_strings(entry):
    entry["generated_answer"] = [ans for ans in entry["generated_answer"] if ans != ""]

def generate_extra_answers(input_file,output_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    for index, entry in enumerate(data):
        if index > 1000:
            break
        print(index)
        human_answers = entry.get("human_answer")

        chatgpt_answers = entry.get("generated_answer")

        generated_chatgpt_answers = [paraphrase(decode(ans)) for ans in human_answers]

        chatgpt_answers.extend(generated_chatgpt_answers)
        entry["generated_answer"] = chatgpt_answers
        remove_empty_strings(entry)


    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)


generate_extra_answers('chatgptSmallerTrainingSet.json', 'test.json')