from transformers import BertModel,BertTokenizer

model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)

print(tokenizer.tokenize('I have a good time, thank you.'))


print('load bert model over')

