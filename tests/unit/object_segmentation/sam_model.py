class sam3_model:
    def __init__(self, engine_file_path, config, overlay_config):
        self.engine_file_path = engine_file_path
        self.config = config
        self.overlay_config = overlay_config
        self.loaded = False
        self.attention_region_calls = []
        self.inference_calls = []
        self.output_override = None

    def load(self):
        self.loaded = True

    def update_attention_region(self, ref_points, frame_shape):
        self.attention_region_calls.append((list(ref_points), frame_shape))

    def __call__(self, frame):
        self.inference_calls.append(frame.shape)
        if self.output_override is not None:
            return self.output_override
        return frame
