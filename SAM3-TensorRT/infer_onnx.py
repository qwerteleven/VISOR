import numpy as np
import cv2
import onnxruntime as ort
from utils.image_preprocess import preprocess_image
from utils.simplify_tokenizer import SimpleCLIPBPETokenizer
from utils.detect_postprocess import process_sam3_results, draw_sam3_results
import json


if __name__ == "__main__":
    with open("config.json") as f:
        config = json.load(f)

    config_metadata = config["metadata"]
    config = config["infer_onnx"]



    image_url = config["image_url"]
    onnx_file_path = config["onnx_file_path"]

    image = cv2.imread(image_url)

    session = ort.InferenceSession(
        onnx_file_path,
        providers=config["providers"]
    )

    vocab_file = config_metadata["vocab_file"]
    merges_file = config_metadata["merges_file"]
    prompt = config["prompt"]

    tokenizer = SimpleCLIPBPETokenizer(
        vocab_file=vocab_file,
        merges_file=merges_file,
        max_length=32,
        bos_token_id=49406,
        eos_token_id=49407,
        bpe_vocab_size=49152,
    )


    processed_image = preprocess_image(image)
    
    ids, mask = tokenizer.encode(prompt)
    input_ids = np.array(ids, dtype=np.int64).reshape(1, -1)
    attention_mask = np.array(mask, dtype=np.int64).reshape(1, -1)

    input_dict = {
        "pixel_values": processed_image.astype("float32"),  
        "input_ids": input_ids,                          
        "attention_mask": attention_mask,              
    }

    outputs = session.run(None, input_dict)

    """
        dimension problem, between 
        outputs[1] pred_boxes

        expected 200, obtain 1

    """ 

    raise Exception("unmatched dimension, inference-postprocess, working in progress")


    image = np.array(image)
    results = process_sam3_results(
        outputs,
        img_h=image.shape[0],
        img_w=image.shape[1],
        score_thr=config["score_thr"],
        mask_thr=config["mask_thr"],
        max_inst=config["max_inst"],
        boxes_normalized=True,
    )
    vis_img = draw_sam3_results(image, results)
    
    cv2.imwrite(config["output_image"], vis_img)
