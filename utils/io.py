import json
import sys
import logging
import datetime as dt
import time
import os
from typing import Dict
import traceback


def load_config(path: str = "config.json") -> Dict:
    """
    
        Load all the general config

    Returns:
        Dict: load config
    """    

    if not os.path.isfile(path):
        msg = f"config file not exists: {path}"
        logging.error(msg)
        print(msg)
        raise Exception(msg)
    

    with open(path, "r") as f:
        try :
            data = json.load(f)
        except json.JSONDecodeError as e:
            msg = f"invalid json format: {e}"
            logging.error(msg)
            print(msg)
            print(traceback.format_exc())

        return data
    

    
def get_config(path: str, section: str) -> Dict:
    """
    
        Get section of configuration in JSON format

    Args:
        path (str): path to the file of configuration
        section (str): section key in configuration

    Raises:
        Exception: check if the file exists
        Exception: check if the section exits
        Exception: check if the json have valid format

    Returns:
        Dict: return the section of configuration
    """    

    if not os.path.isfile(path):
        msg = f"config file not exists: {path}"
        logging.error(msg)
        print(msg)
        raise Exception(msg)
    
    try:
        with open(path) as f:
            try:
                config = json.load(f)
            except json.JSONDecodeError as e:
                msg = f"invalid json format: {e}"
                logging.error(msg)
                print(msg)
                print(traceback.format_exc())

    except FileNotFoundError:
        msg = f"Can not load config json: {path}"
        logging.error(msg)
        print(msg)


    if not section in config:
        msg = f"section config not exists: {path}"
        logging.error(msg)
        print(msg)
        raise Exception(msg)

    config = config[section]

    return config


def oldest_file_in_tree(root_folder: str, extension: str=".log") -> str:
    """
    
        find the oldest file in folder

    Args:
        root_folder (str): path to folder to process
        extension (str, optional): filter files by ext. Defaults to ".log".

    Raises:
        Exception: check if the folder exists

    Returns:
        str: oldest file in folder
    """    

    if not os.path.isdir(root_folder):
        msg = f"folder not exists: {root_folder}"
        logging.error(msg)
        print(msg)
        raise Exception(msg)

    return min(
        (os.path.join(dirname, filename)
        for dirname, dirnames, filenames in os.walk(root_folder)
        for filename in filenames
        if filename.endswith(extension)), key=lambda fn: os.stat(fn).st_mtime)


def set_logger(LOG_PATH: str, service_name: str) -> None:
    """
    
        Iniciates a logger to code flux

    Args:
        LOG_PATH (str): folder to stores the logs
        service_name (str): name of the service to track

    Raises:
        Exception: check if the folder exists
        Exception: check if the folder have more logs that MAX_LOGS

    """    

    config = get_config("../config.json", "logs") 

    timestamp = dt.datetime.fromtimestamp(time.time())

    assert timestamp.timestamp() > 0
    assert timestamp.timestamp() < sys.float_info.max - 1

    date = timestamp.strftime(config["time_format"])

    if not os.path.isdir(LOG_PATH):
        msg = f"logs folder not exists: {LOG_PATH}"
        logging.error(msg)
        print(msg)
        raise Exception(msg)
    
    n_files_in_logs = len([name for name in os.listdir(LOG_PATH) if os.path.isfile(f"{LOG_PATH}/{name}")])
    

    if n_files_in_logs > config["MAX_LOGS"]:
        oldest_log = oldest_file_in_tree(LOG_PATH)
        try:
            os.remove(oldest_log)
        except OSError:
            msg = f"MAX_LOGS files reached, fail to remove the oldest: {LOG_PATH}"
            logging.error(msg)
            print(msg)
            raise Exception(msg)


    LOG_FILE = f"{LOG_PATH}/{service_name}_{date}.log"
    logFormatter = logging.Formatter(config["log_format"])
    fileHandler = logging.FileHandler("{0}".format(LOG_FILE))
    fileHandler.setFormatter(logFormatter)
    rootLogger = logging.getLogger()
    rootLogger.addHandler(fileHandler)
    rootLogger.setLevel(logging.INFO)


