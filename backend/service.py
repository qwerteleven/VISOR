import os
import sys
from proxy import create_app
import uvicorn

root_folder = os.path.abspath(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.append(root_folder)

from utils.io import load_config, set_logger  # noqa: E402

set_logger("../logs", os.path.basename(sys.argv[0]))
config_cache = load_config("../config.json")
app = create_app(config_cache, static_directory="../frontend")

if __name__ == "__main__":
    uvicorn.run(
        app,
        host=config_cache["proxy_config"]["host"],
        port=config_cache["proxy_config"]["port"],
        ssl_keyfile=config_cache["keyfile"],
        ssl_certfile=config_cache["certified"],
        ws=config_cache["proxy_config"]["ws"],
        http=config_cache["proxy_config"]["http"],
        loop=config_cache["proxy_config"]["loop"],
    )
