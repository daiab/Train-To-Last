from utils.test_frame import Param

class MSParam(Param):

    def __init__(self):
        Param.__init__(self)
        self.data_file_path=None
        self.output_file_path=""

    def _parse_file(self, data_type, exts):
        pass


