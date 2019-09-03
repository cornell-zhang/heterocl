
class device(object):
    def __init__(self, name):
        self.name = name
    def __str__(self):
        return self.name
    def __repr__(self):
        return self.__str__()

class cpu(device):
    def __init__(self):
        super(cpu, self).__init__("cpu")

class fpga(device)
    def __init__(self):
        super(cpu, self).__init__("fpga")
