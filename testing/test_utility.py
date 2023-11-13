import sys
import os
sys.path.append("../")
import utility
sys.path.pop()
import json
import numpy as np
import logging

with open("../configuration/pipeline_configuration.json", "r") as fd:
    config = json.load(fd)

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.DEBUG, filename='test_logs.log')
logger = logging.getLogger(__name__)

def test_read_files():
    try:
        utility.read_file(config["data_directory_path"] + config["train_dataset_name"], logger, sheet_name = 0, usecols = None, concat_sheets = False)
        assert True
    except:
        assert False

def test_data_type_conversion():
    data = utility.read_file(config["data_directory_path"] + config["train_dataset_name"], logger, sheet_name = 0, usecols = None, concat_sheets = False)
    try:
        utility.data_type_conversion(data, logger, config["type_conversion_mpping"])
        assert True
    except:
        assert False

def test_string_encoding():
    data = utility.read_file(config["data_directory_path"] + config["train_dataset_name"], logger, sheet_name = 0, usecols = None, concat_sheets = False)
    try:
        utility.string_encoding(data, logger, config["string_columns"])
        assert True
    except:
        assert False

def test_missing_data_imputation():
    data = utility.read_file(config["data_directory_path"] + config["train_dataset_name"], logger, sheet_name = 0, usecols = None, concat_sheets = False)
    try:
        utility.missing_data_imputation(data, logger, "Age", method = "median")
        assert True
    except:
        assert False

def test_create_quantile_feature():
    data = utility.read_file(config["data_directory_path"] + config["train_dataset_name"], logger, sheet_name = 0, usecols = None, concat_sheets = False)
    try:
        utility.create_quantile_feature(data, logger, "Age", q = 10)
        assert True
    except:
        assert False

def test_normalize():
    data = utility.read_file(config["data_directory_path"] + config["train_dataset_name"], logger, sheet_name = 0, usecols = None, concat_sheets = False)
    try:
        utility.normalize(data, logger, config["data_directory_path"] + config["scaler_file_name"], fit_standard_scaler = True)
        assert True
    except:
        assert False
