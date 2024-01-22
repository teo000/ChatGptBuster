import random
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt
import json

def plot_results(history):
    history_dict = history.history
    history_dict.keys()

    acc = history_dict['accuracy']
    val_acc = history_dict['val_accuracy']
    loss = history_dict['loss']
    val_loss = history_dict['val_loss']

    epochs = range(1, len(acc) + 1)

    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.show()

    plt.clf()

    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.show()


with open('cleanchatgpt.json', 'r') as file:
    data = json.load(file)

questions = [entry['question'] for entry in data]
answers = [entry['answer'] for entry in data]
labels = [entry['label'] for entry in data]

instances = [question + " " + answer for question, answer in zip(questions, answers)]
instances_with_labels = list(zip(instances, labels))

random.shuffle(instances_with_labels)

split_index = int(0.8 * len(instances_with_labels))
train_data = instances_with_labels[:split_index]
test_data = instances_with_labels[split_index:]

train_examples, train_labels = zip(*train_data)
test_examples, test_labels = zip(*test_data)

model = "https://tfhub.dev/google/nnlm-en-dim50/2"
hub_layer = hub.KerasLayer(model, dtype=tf.string, trainable=True)

model = tf.keras.Sequential()
model.add(hub_layer)
model.add(tf.keras.layers.Dense(16, activation='relu'))
model.add(tf.keras.layers.Dense(1))


model.compile(optimizer='adam',
              loss=tf.losses.BinaryCrossentropy(from_logits=True),
              metrics=[tf.metrics.BinaryAccuracy(threshold=0.0, name='accuracy')])

num_train_examples = len(train_examples)
x_val = train_examples[:num_train_examples//2]
x_train = train_examples[num_train_examples // 2:]

y_val = train_labels[:num_train_examples//2]
y_train = train_labels[num_train_examples // 2:]

history = model.fit(x_train,
                    y_train,
                    epochs=50,
                    batch_size=1024,
                    validation_data=(x_val, y_val),
                    verbose=1)

model.save("chatgpt_buster_model")

predictions = model.predict(test_examples)
print(predictions)

predictions = model.predict(test_examples)
predicted_labels = (predictions > 0).astype(int).flatten()
for i in range(len(test_examples)):
    print(f"Example {i + 1}:")
    print("Test Example:", test_examples[i])
    print("Real Value:", test_labels[i])
    print("Predicted Value:", predicted_labels[i])
    print("--------------------")

results = model.evaluate(test_examples, test_labels)
print(results)

plot_results(history)
