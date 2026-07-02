from transformers import AutoProcessor, AutoModelForMultimodalLM
import time

start = time.time()


MODEL_ID = "google/gemma-4-E4B-it"

# Load model
processor = AutoProcessor.from_pretrained(MODEL_ID)
model = AutoModelForMultimodalLM.from_pretrained(
    MODEL_ID, 
    dtype="auto", 
    device_map="auto"
)


for name, buf in model.named_buffers():
    if "input_min" in name or "input_max" in name or "output_min" in name or "output_max" in name:
        print(name, buf.item() if buf.numel() == 1 else buf.shape, buf)

exit()

# Prompt - add image before text
messages = [
    {
        "role": "user", "content": [
            {"type": "image", "image": "http://images.cocodataset.org/val2017/000000077595.jpg"},
            {"type": "text", "text": "Generate a description of the primary object, respond to this questions, whats is?, for what work, about what is compose?"}
        ]
    }
]

# Process input
inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
    add_generation_prompt=True,
).to(model.device)

print(inputs["pixel_values"].shape)

exit()
input_len = inputs["input_ids"].shape[-1]

# Generate output
outputs = model.generate(**inputs, max_new_tokens=512)
response = processor.decode(outputs[0][input_len:], skip_special_tokens=False)

# Parse output

response = processor.parse_response(response)

print(response)

stop = time.time()


print(f"Total time: {stop - start}s")