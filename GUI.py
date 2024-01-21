import tkinter as tk
from tkinter import messagebox
import random
import json
from paraphrase import paraphrase
import tensorflow as tf

model = tf.keras.models.load_model("chatgpt_buster_model")
def get_chatgpt_answer(question, answer1, answer2):
    answer1_full, answer2_full = question + " " + answer1, question + " " + answer2

    results = model.predict([answer1_full, answer2_full])
    probabilities = tf.nn.sigmoid(results).numpy()
    print(probabilities)
    if results[0][0] > results[1][0]:
        return answer1, answer2
    else:
        return answer2, answer1

class QuestionGeneratorApp:
    def __init__(self, root, questions):
        self.root = root
        self.root.title("Random Question Generator")

        self.root.geometry("800x600")

        self.root.configure(bg='#FF69B4')

        self.questions = questions
        self.current_question_index = None

        self.question_label = tk.Label(root, text="", bg='#FF69B4', font=("Arial", 14), anchor="center", wraplength=700)
        self.question_label.pack(pady=10)

        self.generated_answer_label = tk.Label(root, text="", bg='#FF69B4', font=("Arial", 14), anchor="center",
                                               wraplength=700)
        self.generated_answer_label.pack(pady=10)

        self.random_question_button = tk.Button(root, text="Random Question", command=self.generate_random_question)
        self.random_question_button.pack(pady=10)

        self.answer_entry = tk.Text(root, font=("Arial", 12), wrap=tk.WORD, height=5)
        self.answer_entry.pack(pady=10)

        self.submit_button = tk.Button(root, text="Submit Answer", command=self.submit_answer)
        self.submit_button.pack(pady=10)

        self.center_window()

    def center_window(self):
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()

        x = (screen_width - 800) // 2
        y = (screen_height - 600) // 2

        self.root.geometry(f"800x600+{x}+{y}")

    def generate_random_question(self):
        if self.questions:
            self.current_question_index = random.randint(0, len(self.questions) - 1)
            random_question = self.questions[self.current_question_index]
            self.question_label.config(text=random_question)
            self.generated_answer_label.config(text="")
        else:
            self.question_label.config(text="No questions available.")

    def submit_answer(self):
        if self.current_question_index is not None:
            answer = self.answer_entry.get("1.0", tk.END)

            modified_answer = paraphrase(answer)
            self.generated_answer_label.config(text=f"Generated Answer: {modified_answer}")

            question = self.questions[self.current_question_index]

            generated_answer, human_answer = self.predict(question, answer, modified_answer)

            message = f"Question: {question}\nUser Answer: {human_answer}\nModified Answer: {generated_answer}"
            messagebox.showinfo("Result", message)

        else:
            messagebox.showinfo("No Question", "Please generate a question first.")

    def predict(self, question, user_answer, modified_answer):
        chatgpt_answer, human_answer = get_chatgpt_answer(question, user_answer, modified_answer)
        print(f"Question: {question}, User Answer: {human_answer}, Generated Answer: {chatgpt_answer}")
        return chatgpt_answer, human_answer


def load_questions_from_json(json_file):
    try:
        with open(json_file, "r", encoding='utf-8') as file:
            data = json.load(file)
            return [entry["question"] for entry in data]
    except FileNotFoundError:
        return []



def decode(str):
    return str.encode('utf-8').decode('unicode_escape')


if __name__ == "__main__":
    questions = load_questions_from_json("cleanchatgpt.json")

    if not questions:
        print("No questions found. Please check your JSON file.")

    root = tk.Tk()
    app = QuestionGeneratorApp(root, questions)
    root.mainloop()
