from base import *


class PipelineManager:
    def __init__(self, input_data_source, model):
        self.input_queue = Queue(maxsize=10)
        self.middle_queue = Queue(maxsize=10)
        self.output_queue = Queue(maxsize=10)

        self.data_preloader = DataPreloader(input_data_source, self.input_queue)
        self.model_inferencer = ModelInferencer(model, self.input_queue, self.middle_queue)
        self.post_processor = PostProcessor(self.middle_queue, self.output_queue)

        self.stages = [self.data_preloader, self.model_inferencer, self.post_processor]

    def start(self):
        """Start all stages."""
        for stage in self.stages:
            stage.start()

    def stop(self):
        """Stop all stages."""
        for stage in self.stages:
            stage.stop()

    def wait_for_completion(self):
        """Wait for all queues to be processed."""
        self.input_queue.join()
        self.middle_queue.join()
        self.output_queue.join()