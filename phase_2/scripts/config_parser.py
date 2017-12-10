import configparser
import logging

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


class ParseConfig(object):
    def __init__(self):
        self.config = configparser.ConfigParser()
        self.config.read("..\config.ini")

    def config_section_mapper(self, section):  # Code to parse a section of the config.ini file
        dict1 = {}
        options = self.config.options(section)
        for option in options:
            dict1[option] = self.config.get(section, option)

        return dict1


if __name__ == "__main__":
    parsed_config = ParseConfig()
    log.info("Parsed config file")
    log.info(parsed_config.config_section_mapper("filePath"))
