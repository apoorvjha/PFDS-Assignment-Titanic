import pandas as pd
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pickle

def read_file(path, logger, sheet_name = 0, usecols = None, concat_sheets = False):
    file_extension = path.split('.')[-1]
    logger.info(f"The file extension is {file_extension}!")
    if file_extension == 'csv':
        try:
            data = pd.read_csv(path)
            logger.info(f"The file at path {path} read successfully with {data.shape[0]} rows and {data.shape[1]} columns!")
        except Exception as e:
            logger.error(f"The read of file at path {path} failed due to {e}!")
    elif file_extension == 'xlsx':
        try:
            data = pd.read_excel(path, sheet_name = sheet_name, usecols = usecols)
            logger.info(f"The file at path {path} read successfully with {data.shape[0]} rows and {data.shape[1]} columns!")
            if concat_sheets == True:
                concated_data = pd.DataFrame()
                for key in data:
                    concated_data = pd.concat([concated_data, data[key]] , axis = 0, ignore_index = True)
            data = concated_data.copy()
        except Exception as e:
            logger.error(f"The read of file at path {path} failed due to {e}!")
    elif file_extension == 'json':
        try:
            with open(path, "r") as fd:
                data = json.load(fd)
            logger.error(f"The file at path {path} read successfully!")
        except Exception as e:
            logger.error(f"The read of file at path {path} failed due to {e}!")
    else:
        logger.error(f"The extension {file_extension} is not supported!")
        data = pd.DataFrame()
    return data

def data_type_conversion(data, logger, mapping):
    logger.info(f"Explicitly converting the data type of columns leveraging {mapping}")
    for column in mapping.keys():
        try:
            data[column] = data[column].astype(mapping[column])
            logger.info(f"Data type conversion for {column} into {mapping[column]} completed successfully!")
        except Exception as e:
            logger.error(f"Data type conversion for {column} into {mapping[column]} Failed due to {e}!")
    return data

def string_encoding(data, logger, string_columns):
    logger.info(f"converting the string columns into neumeric features!")
    try:
        data = pd.get_dummies(data, columns = string_columns)
        logger.info(f"convertion of string columns {string_columns} into neumeric features completed successfully!")
    except Exception as e:
        logger.error(f"convertion of string columns {string_columns} into neumeric features failed due to {e}!")
    return data

def missing_data_imputation(data, logger, column, method = "median"):
    assert method in ['median', 'mean', 'mode'], "Only 'median', 'mean' and 'mode' inputation methods are supported!"
    logger.info(f"Imputing missing values for {column} with {method}!")
    if method == 'mean':
        try:
            data[column].fillna(data[column].mean(), inplace = True)
            logger.info(f"Missing value imputation for {column} completed successfully!")
        except Exception as e:
            logger.info(f"Missing value imputation for {column} failed due to {e}!")
    elif method == 'median':
        try:
            data[column].fillna(data[column].median(skipna = True), inplace = True)
            logger.info(f"Missing value imputation for {column} completed successfully!")
        except Exception as e:
            logger.info(f"Missing value imputation for {column} failed due to {e}!")
    else:
        try:
            data[column].fillna(data[column].mode(dropna = True)[0], inplace = True)
            logger.info(f"Missing value imputation for {column} completed successfully!")
        except Exception as e:
            logger.info(f"Missing value imputation for {column} failed due to {e}!")
    return data

def create_quantile_feature(data, logger, column, q = 10):
    logger.info(f"Creating quantile feature with q = {q} for {column} column!")
    try:
        data[column + f"_{q}_quantile_feature"] = pd.qcut(data[column], q = q, labels = False, retbins = False).tolist()
        logger.info(f"Created quantile feature with q = {q} for {column} column successfully!")
    except Exception as e:
        logger.info(f"quantile feature with q = {q} for {column} column creation failed due to {e}!")
    return data

def split_train_validation(data, logger, features, target):
    X_train, X_val, Y_train, Y_val = train_test_split(data[features] , data[target], test_size=0.2,random_state=45,shuffle=True)
    return pd.DataFrame(data = X_train, columns = features), pd.DataFrame(data = X_val, columns = features), Y_train, Y_val

def normalize(data, logger, scaler_path, fit_standard_scaler = True):
    if fit_standard_scaler == True:
        scaler = StandardScaler()
        data = scaler.fit_transform(data)
        data = pd.DataFrame(data = data, columns = scaler.feature_names_in_)
        with open(scaler_path, "wb") as fd:
            pickle.dump(scaler, fd)
    else:
        with open(scaler_path, "rb") as fd:
            scaler = pickle.load(fd)
        data = scaler.transform(data)
        data = pd.DataFrame(data = data, columns = scaler.feature_names_in_)
    return data