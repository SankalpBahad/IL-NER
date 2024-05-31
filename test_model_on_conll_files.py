from datasets import Dataset
import torch
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForTokenClassification
from seqeval.metrics import classification_report, f1_score
from argparse import ArgumentParser

def convert_conll_to_dataset(file_path):
    sentences = []
    true_labels = []
    ids=[]

    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        sentence = []
        label = []
        id_=0

        for line in lines:
            line = line.strip()
            if line == '':
                if sentence:
                    sentences.append(sentence)
                    true_labels.append(label)
                    ids.append(id_)
                    sentence = []
                    label = []
                    id_+=1
            else:
                parts = line.split('\t')
                token = parts[0]
                if len(parts[-1])==1:
                    tag=parts[-1]
                else:
                    tag=parts[-1]
                sentence.append(token)
                if tag in label_list:
                    label.append(label_to_id[tag])
                else:
                    label.append(6)
    if sentence:
        sentences.append(sentence)
        true_labels.append(label)
        ids.append(id_)
    data = {"id":ids,"tokens": sentences, "ner_tags": true_labels}
    dataset = Dataset.from_dict(data)
    return dataset

def predict_labels_for_sentences(model, tokenizer, sentences):
    predicted_true_labels_for_all_sents = []
    with torch.no_grad():
        for index, sentence in enumerate(sentences):
            input_tensors = tokenizer(
                sentence, is_split_into_words=True,truncation=True, return_tensors='pt', max_length=512)
            outputs = model(**input_tensors)
            logit_values = outputs.logits
            arg_max_torch = torch.argmax(logit_values, axis=-1)
            predicted_tokens_classes = [
                model.config.id2label[t.item()] for t in arg_max_torch[0]]
            word_ids = input_tensors.word_ids()
            previous_word_idx = 0
            label_ids = []
            for word_index in range(len(word_ids)):
                if word_ids[word_index] == None:
                    previous_word_idx = word_ids[word_index]
                elif word_ids[word_index] != previous_word_idx:
                    label_ids.append(predicted_tokens_classes[word_index])
                    previous_word_idx = word_ids[word_index]
                else:
                    previous_word_idx = word_ids[word_index]
            predicted_true_labels_for_all_sents.append(label_ids)
    return predicted_true_labels_for_all_sents

# # Use a pipeline as a high-level helper
# pipe = pipeline("token-classification", model="Model_Name")

# Load model directly
from transformers import AutoTokenizer, AutoModelForTokenClassification

tokenizer = AutoTokenizer.from_pretrained("Model_Name")
model = AutoModelForTokenClassification.from_pretrained("Model_Name")

label_list = ['B-NEL','B-NEO','B-NEP','I-NEL','I-NEO','I-NEP','O','B-NETI','I-NETI','B-NEN','I-NEN','B-NEAR','I-NEAR']
label_to_id = {label_list[i] : i for i in range(len(label_list))}

parser = ArgumentParser()
parser.add_argument('--file', dest='file', help='Enter the test file')
args = parser.parse_args()
print(args.file)
test_file_name=args.file
dataset=convert_conll_to_dataset(test_file_name)
sentences=[]
true_labels=[]
tokens=[]
for j in dataset:
    sentence=j["tokens"]
    tokens.append(j["tokens"])
    sentences.append(sentence)
    tags=[label_list[i] for i in j["ner_tags"]]
    true_labels.append(tags)

predicted_labels=predict_labels_for_sentences(model, tokenizer, sentences)

print(f1_score(true_labels,predicted_labels))
