import pandas as pd
import json


class Json:
    def __init__(self) -> None:
        pass

    def count_elements_in_json(self, json_file_path):
        with open(json_file_path, "r") as json_file:
            data = json.load(json_file)

            # Check if the data is a list or a dictionary
            if isinstance(data, list):
                return len(data)
            elif isinstance(data, dict):
                return len(data.keys())
            else:
                # If the JSON file has a different top-level structure (e.g., int, string), return 1
                return 1
