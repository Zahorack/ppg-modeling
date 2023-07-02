
class Processor:
    """
    Interface for all processor classes
    """
    def process(self, data):
        raise NotImplementedError


class Pipeline:
    """
    Pipeline class to chain processor classes
    """
    def __init__(self, processors):
        self.processors = processors

    def process(self, data):
        for processor in self.processors:
            data = processor.process(data)
        return data
