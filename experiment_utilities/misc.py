# Turns a dictionary into a class
class Dict2Class(object):
    def __init__(self, d):
        for key in d:
            setattr(self, key, d[key])