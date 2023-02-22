from abc import ABC, abstractclassmethod

class FileReader(ABC):
    @abstractclassmethod
    def read(self):
        pass

    @abstractclassmethod
    def close(self):
        pass

