import json

class DataAccess:
    def __init__(self, file_path=None):
        self.file_path = file_path

    def load_data(self, file_path=None):
        if file_path is None and self.file_path is None:
            raise ValueError("È necessario fornire un percorso di file per caricare i dati.")

        file_path = file_path or self.file_path

        with open(file_path, 'r') as file:
            data = json.load(file)

        return data
