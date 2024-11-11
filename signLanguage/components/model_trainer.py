import os
import sys
import yaml  # type: ignore
import shutil
import zipfile
import subprocess
from signLanguage.utils.main_utils import read_yaml_file
from signLanguage.logger import logging
from signLanguage.exception import SignException
from signLanguage.entity.config_entity import ModelTrainerConfig
from signLanguage.entity.artifacts_entity import ModelTrainerArtifact

class ModelTrainer:
    def __init__(self, model_trainer_config: ModelTrainerConfig):
        self.model_trainer_config = model_trainer_config

    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        logging.info("Starting initiate_model_trainer method of ModelTrainer class")

        try:
            # Step 1: Ensure the zip file exists before unzipping
            zip_file = "suresh.v1i.yolov5pytorch.zip"
            if not os.path.exists(zip_file):
                raise FileNotFoundError(f"Required file '{zip_file}' not found for training.")
            
            logging.info("Unzipping training data.")
            # Unzipping the file using Python's zipfile module for better control
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                zip_ref.extractall()  # Extract all contents in the current directory

            # Remove the zip file after extraction
            os.remove(zip_file)

            # Step 2: Read number of classes from data.yaml
            data_file = r'D:\sign-language-detection-\artifacts\11_06_2024_16_31_02\data_ingestion\feature_store\data.yaml'
            
            if not os.path.exists(data_file):
                raise FileNotFoundError(f"Required configuration file '{data_file}' not found.")

            logging.info("Reading data.yaml file.")
            try:
                with open(data_file, 'r') as stream:
                    num_classes = yaml.safe_load(stream).get('nc')
                    if num_classes is None:
                        raise ValueError("Missing 'nc' key in data.yaml.")
            except yaml.YAMLError as e:
                raise ValueError(f"Error reading the YAML file: {e}")

            logging.info(f"Number of classes for training: {num_classes}")

            # Step 3: Update model configuration
            model_config_file_name = self.model_trainer_config.weight_name.split(".")[0]
            config_path = os.path.join("yolov5", "models", f"{model_config_file_name}.yaml")
            custom_config_path = os.path.join("yolov5", "models", f"custom_{model_config_file_name}.yaml")

            if not os.path.exists(config_path):
                raise FileNotFoundError(f"Model configuration file '{config_path}' not found.")

            config = read_yaml_file(config_path)
            config['nc'] = int(num_classes)

            with open(custom_config_path, 'w') as f:
                yaml.dump(config, f)
            logging.info(f"Updated model configuration saved to {custom_config_path}")

            # Step 4: Start model training
            logging.info("Starting model training with YOLOv5.")
            train_command = (
                f"cd yolov5/ && python train.py --img 416 --batch {self.model_trainer_config.batch_size} "
                f"--epochs {self.model_trainer_config.no_epochs} --data ../{data_file} "
                f"--cfg ./models/custom_{model_config_file_name}.yaml --weights {self.model_trainer_config.weight_name} "
                f"--name yolov5s_results --cache"
            )
            
            # Run training command and capture output
            result = subprocess.run(train_command, shell=True, capture_output=True, text=True)
            logging.info(f"Training Output: {result.stdout}")
            logging.error(f"Training Error: {result.stderr}")

            if result.returncode != 0:
                raise SignException(f"Training failed, check the logs for errors: {result.stderr}", sys)

            # Step 5: Check for trained model
            trained_model_path = os.path.join("yolov5", "runs", "train", "yolov5s_results", "weights", "best.pt")
            logging.info(f"Checking if trained model exists at {trained_model_path}")
            
            if not os.path.exists(trained_model_path):
                logging.error(f"Trained model file not found at {trained_model_path}")
                raise FileNotFoundError(f"Trained model file not found at {trained_model_path}")

            # Step 6: Copy trained model to destination
            os.makedirs(self.model_trainer_config.model_trainer_dir, exist_ok=True)
            shutil.copy(trained_model_path, "yolov5/")
            shutil.copy(trained_model_path, self.model_trainer_config.model_trainer_dir)

            # Step 7: Cleanup
            logging.info("Cleaning up temporary files.")
            shutil.rmtree("yolov5/runs")  # Recursively remove the runs directory
            shutil.rmtree("train")  # Remove train folder if no longer needed
            shutil.rmtree("test")  # Remove test folder if no longer needed
            if os.path.exists(data_file):
                os.remove(data_file)

            # Step 8: Prepare artifact
            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_file_path="yolov5/best.pt"
            )
            logging.info("Completed model training successfully.")
            logging.info(f"Model trainer artifact: {model_trainer_artifact}")

            return model_trainer_artifact

        except FileNotFoundError as e:
            logging.error(f"File error: {e}")
            raise SignException(e, sys)

        except ValueError as e:
            logging.error(f"Value error: {e}")
            raise SignException(e, sys)

        except Exception as e:
            logging.error(f"An unexpected error occurred: {e}")
            raise SignException(e, sys)
