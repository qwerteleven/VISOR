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


# Prompt - add image before text
messages = [
    {
        "role": "user", "content": [
            {"type": "image", "image": "../SAM3-TensorRT/image_test/person_test.jpeg"},
            {"type": "text", "text": "What is shown in this image?"}
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
input_len = inputs["input_ids"].shape[-1]

# Generate output
outputs = model.generate(**inputs, max_new_tokens=512)
response = processor.decode(outputs[0][input_len:], skip_special_tokens=False)

# Parse output

response = processor.parse_response(response)

print(response)

stop = time.time()


print(f"Total time: {stop - start}s")