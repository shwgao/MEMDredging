from abc import ABC, abstractmethod
from threading import Thread
from queue import Queue

class PipelineStage(ABC):
    """An abstract base class representing a stage in a processing pipeline. I plan to make three stages:
    DataPreloader, ModelInferencer, and PostProcessor to inherit from this class. We can add some syncronization
    mechanism to make sure the pipeline is working as expected.
    """
    def __init__(self, input_queue=None, output_queue=None):
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.thread = Thread(target=self.run)
        self.stop_signal = False
        self.load_data_signal = False

    @abstractmethod
    def process(self, data):
        """Abstract method to define the processing logic."""
        pass

    def run(self):
        """Continuously process data from the input queue."""
        while not self.stop_signal:
            try:
                data = self.input_queue.get(timeout=1)  # Timeout to check stop_signal
                if data is None:  # End of pipeline signal
                    self.stop()
                    break
                processed_data = self.process(data)
                if self.output_queue:
                    self.output_queue.put(processed_data)
                self.input_queue.task_done()
            except Exception as e:
                continue

    def start(self):
        """Start the thread."""
        self.thread.start()

    def stop(self):
        """Stop the thread."""
        self.stop_signal = True
        self.thread.join()


class DataPreloader(PipelineStage):
    def __init__(self, input_data_source, device='cuda'):
        """
        Args:
            input_data_source: a pytorch dataloader
            device: torch device to load data to (default: 'cuda')
        """
        super().__init__(input_queue=Queue())
        self.input_data_source = input_data_source
        self.device = device
        self.data_iterator = iter(input_data_source)
        self.num_batches = len(input_data_source)
        
    def process(self):
        """Load and preprocess data.
        
        Returns:
            Preprocessed data on specified device
        """
        try:
            # Get next batch from dataloader
            batch = next(self.data_iterator)
            self.num_batches -= 1
            return self.load_and_preprocess(batch)
        except StopIteration:
            # Signal end of data
            return None

    def load_and_preprocess(self, data):
        """Move data to specified device.
        
        Args:
            data: tuple/list of torch tensors
        Returns:
            Same data structure with tensors moved to device
        """
        # Handle both single tensors and collections of tensors
        if isinstance(data, (tuple, list)):
            return [item.to(self.device) for item in data]
        return data.to(self.device)
    
    def __len__(self):
        """
        Return the remaining number of batches in the dataloader.
        """
        return self.num_batches


class ModelInferencer(PipelineStage):
    def __init__(self, model, input_queue, output_queue):
        super().__init__(input_queue=input_queue, output_queue=output_queue)
        self.model = model

    def process(self, data):
        """Run the model inference."""
        inference_result = self.model_inference(data)
        return inference_result

    def model_inference(self, data):
        # Simulate model computation
        return f"Inference result for {data}"


class PostProcessor(PipelineStage):
    def process(self, data):
        """Post-process the inference result."""
        postprocessed_result = self.postprocess(data)
        # Simulate returning the result or saving it
        print(f"Post-processed result: {postprocessed_result}")
        return postprocessed_result

    def postprocess(self, data):
        # Simulate post-processing
        return f"Post-processed {data}"


