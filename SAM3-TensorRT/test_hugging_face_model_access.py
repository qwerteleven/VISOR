from huggingface_hub import login
login()


from transformers import pipeline

pipe = pipeline("mask-generation", model="facebook/sam3")  
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("facebook/sam3")
model = AutoModel.from_pretrained("facebook/sam3")