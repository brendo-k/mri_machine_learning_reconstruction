from abc import ABC, abstractclassmethod

class FileReader(ABC):
    @abstractclassmethod
    def __enter__(self):
        pass
 
    @abstractclassmethod
    def __exit__(self):
        pass