import configparser

class Config:
    def __init__(self, filepath):
        self.config = configparser.ConfigParser()
        self.config.read(filepath)

    def get(self, section, option):
        # Restituisce il valore come stringa
        return self.config.get(section, option)

    def get_int(self, section, option):
        # Restituisce il valore come intero
        return self.config.getint(section, option)

    def get_float(self, section, option):
        # Restituisce il valore come float
        return self.config.getfloat(section, option)

    def get_boolean(self, section, option):
        # Restituisce il valore come booleano
        return self.config.getboolean(section, option)

    def get_section(self, section):
        # Restituisce tutti i valori della sezione come un dizionario
        return dict(self.config.items(section))
