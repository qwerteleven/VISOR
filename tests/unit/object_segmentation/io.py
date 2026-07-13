FAKE_CONFIG_SECTIONS = {
    "sam_streaming_quality": {"format": "JPEG", "quality": 80, "fps": 15},
    "sam_api_timeouts": {
        "connect_rtsp": 0.01,
        "stale_connection": 0.01,
        "read_frame": 0.01,
        "disconnect_retry": 0.01,
        "rtsp_retry": 0.01,
        "frame_generator": 0.01,
    },
    "demo_trt_webcam": {"engine_file_path": "fake_engine.trt"},
    "streaming_overlay": {},
    "input_source": "rtsp://fake-source",
    "streaming_mediatype": "multipart/x-mixed-replace; boundary=frame",
    "endpoints": {"stream": "/stream", "overlay": "/ws/overlay"},
}


def get_config(path, section):
    return FAKE_CONFIG_SECTIONS[section]


def set_logger(*args, **kwargs):
    pass
