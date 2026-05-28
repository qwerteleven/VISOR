import numpy as np
import cv2
import json
import tensorrt as trt
import utils.common as common
import time
from utils.image_preprocess import preprocess_image
from utils.simplify_tokenizer import SimpleCLIPBPETokenizer
from utils.detect_postprocess import process_sam3_results, draw_sam3_results


with open("config.json") as f:
    config = json.load(f)

config_metadata = config["metadata"]
config = config["infer_trt_webcam"]


def main_loop():
    
    cummulative_time = 0
    n_iteration = 0 

    for _ in range(config["max_iteration"]):
        n_iteration += 1
        start = time.time()
        ret, image = cam.read()
        
        if not ret:
            print("failed to grab frame")
            break


        processed_image = preprocess_image(image)
        
        ids, mask = tokenizer.encode(prompt)
        input_ids = np.array(ids, dtype=np.int64).reshape(1, -1)
        attention_mask = np.array(mask, dtype=np.int64).reshape(1, -1)


        np.copyto(inputs[0].host, processed_image.ravel())
        np.copyto(inputs[1].host, input_ids.ravel())
        np.copyto(inputs[2].host, attention_mask.ravel())

        output_model = common.do_inference(context, engine=engine, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)

        output_model = [output_model[0].reshape(*config["outputs_shapes"][0]), output_model[1].reshape(*config["outputs_shapes"][1]), output_model[2].reshape(*config["outputs_shapes"][2])]

        image = np.array(image)
        results = process_sam3_results(
            output_model,
            img_h=image.shape[0],
            img_w=image.shape[1],
            score_thr=config["score_thr"],
            mask_thr=config["mask_thr"],
            max_inst=config["max_inst"],
            boxes_normalized=config["boxes_normalized"],
        )

        vis_img = draw_sam3_results(image, results)
        cv2.imshow("test", vis_img)
        cv2.waitKey(1)  # Wait for a key press to close the window

        end = time.time()

        cummulative_time += end-start

        if n_iteration % 100 == 0:

            print(f"Process: time: {cummulative_time /  100}")
            n_iteration = 0 
            cummulative_time = 0




if __name__ == "__main__":

    cam = cv2.VideoCapture(0)
    engine_file_path = config["engine_file_path"]

    logger = trt.Logger(trt.Logger.WARNING)
    with open(engine_file_path, "rb") as f, trt.Runtime(logger) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
    context = engine.create_execution_context()
    inputs, outputs, bindings, stream = common.allocate_buffers(engine)

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
    
    main_loop()

    cv2.destroyAllWindows()  # Close the window
    cam.release()

