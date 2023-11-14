from utility import *
import logging

def main():
    # Instantiate logger instance
    logging.basicConfig(format = '[%(asctime)s] : %(levelname)s -> %(message)s', level = logging.DEBUG, filename = '../logs/pipeline_log.log')
    logger = logging.getLogger(__name__)
    # Read Configuration file
    config = read_file("../configuration/pipeline_configuration.json", logger)
    # Read datasets 
    train_data = read_file(config["data_directory_path"] + config["train_dataset_name"], logger)
    test_data = read_file(config["data_directory_path"] + config["test_dataset_name"], logger)
    # Data Type conversion
    train_data = data_type_conversion(train_data, logger, config["type_conversion_mpping"])
    test_data = data_type_conversion(test_data, logger, config["type_conversion_mpping"])
    # String Encoding
    train_data = string_encoding(train_data, logger, config["string_columns"])
    test_data = string_encoding(test_data, logger, config["string_columns"])
    # Missing data Imputation
    for column in config["imputation_method_mapping"].keys():
        train_data = missing_data_imputation(train_data, logger, column, method = config["imputation_method_mapping"][column])
        test_data = missing_data_imputation(test_data, logger, column, method = config["imputation_method_mapping"][column]) 
    # Feature Engineering -> Creating decile features for Fare and Age to capture the trend.
    for feature_eng_column in config["feature_engineering_columns"]:
        train_data = create_quantile_feature(train_data, logger, feature_eng_column, q = 4)
        test_data = create_quantile_feature(test_data, logger, feature_eng_column, q = 4)
    # Drop unnecessary columns
    train_data_model_data = train_data.drop(columns = config["drop_columns"])
    test_data_model_data = test_data.drop(columns = config["drop_columns"])
    # Train-Validation Split of data
    features = list(set(train_data_model_data.columns).difference(set([config["target_column"]])))
    target = config["target_column"]
    X_train, X_val, Y_train, Y_val = split_train_validation(train_data_model_data, logger, features, target)
    # Normalize / Scale the data for model
    X_train = normalize(X_train[features], logger, config["data_directory_path"] + config["scaler_file_name"], fit_standard_scaler = True)
    X_val = normalize(X_val[features], logger, config["data_directory_path"] + config["scaler_file_name"], fit_standard_scaler = False)
    X_test = normalize(test_data_model_data[features], logger, config["data_directory_path"] + config["scaler_file_name"], fit_standard_scaler = False)
    # Model Iteration Code -> TBU

if __name__ == '__main__':
    main()