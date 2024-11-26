# 管道生成器，通过管理ThreadPredictor类为客户端提供预测接口


from lib.thread_predictor import ThreadPredictor

class PipeGenerator(object):
    def __init__(self, network):
        self.network = network
        self.predictor = None

    def get_pipe(self):
        if self.predictor is None:
            self.predictor = ThreadPredictor(self.network)
            self.predictor.start()
        return self.predictor.create_pipe() 