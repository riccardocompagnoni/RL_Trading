from pyspark import SparkConf
import yaml


class SparkConfWrapper:
    def __init__(self, path: str = "prj/app/config/spark_conf.yaml") -> None:
        self.conf = get_obj_from_config_path(path)
        self.sc = SparkConf()

    def load_conf(self, yaml_key: str):
        for key, value in self.conf.get(yaml_key).items():
            self.sc.set(key, value)

    def add_basic(self):
        self.load_conf("basic")


def get_obj_from_config_path(path:str):
    with open(path, "r") as file:
        dict_config = yaml.safe_load(file)
        return dict_config
