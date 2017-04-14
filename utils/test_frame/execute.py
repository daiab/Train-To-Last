import importlib
import sys
sys.path.append("project root path")

class Execute(object):
    def __init__(self):
        self.project_name=None
        self.project_root_path=None
        module = importlib.import_module("models.nn4")
        self.evaluator=module.evaluator

    def start_eval(self):
        for value in self.evaluator():
            pass

    def write_feature_file(self):
        for value in self.evaluator():
            pass