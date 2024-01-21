import tensorflow as tf
import tensorflow_datasets as tfds


model = tf.keras.models.load_model("chatgpt_buster_model")

test_examples = [
    "Why is it not illegal to belong to a gang ? If law enforcement and the different levels of the government know that gangs engage in criminal activity , why is someone not charged simply for belonging to a criminal organization ? Please explain like I'm five. I am in a gang and I don't feel guilty about it, crime is awesome !!!!!!",
    "Why is it not illegal to belong to a gang ? If law enforcement and the different levels of the government know that gangs engage in criminal activity , why is someone not charged simply for belonging to a criminal organization ? Please explain like I'm five. I am affiliated with a group, and I don't experience remorse for it; engaging in illicit activities is thrilling!"
]

results = model.predict(test_examples)
print(results)

