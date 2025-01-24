from base import *   
import threading
import queue
import torch
from torch.utils.data import Dataset, DataLoader as TorchDataLoader
import random

class InferenceDataset(Dataset):
    def __init__(self, size=1000):
        self.data = torch.randn(size, 3, 224, 224)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

class BasePipelineStage:
    def __init__(self):
        self._stop_event = threading.Event()
        self.thread = None

    def start(self):
        """Start the stage's thread"""
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        return self.thread

    def stop(self):
        """Signal the stage to stop"""
        self._stop_event.set()

    def _run(self):
        """Main execution loop to be implemented by subclasses"""
        raise NotImplementedError

class DataLoader(BasePipelineStage):
    def __init__(self, dataset, task_queue, batch_size=32):
        super().__init__()
        self.loader = TorchDataLoader(dataset, batch_size=batch_size)
        self.task_queue = task_queue
        self.stream = torch.cuda.Stream()
        self.current_batch = None
        self.counter = 0
        self.has_more_data = True
        
    def trigger_load(self):
        if not self.has_more_data:
            return False
        if not self._stop_event.is_set() and self.task_queue.qsize() < 3:
            with torch.cuda.stream(self.stream):
                try:
                    batch = next(self.loader)
                    self.current_batch = batch.cuda(non_blocking=True)
                except StopIteration:
                    self.has_more_data = False
                    self._stop_event.set()
                    return False
            return True
        return False

    def complete_load(self):
        if self.current_batch is not None:
            torch.cuda.current_stream().wait_stream(self.stream)
            self.task_queue.put(('data', self.current_batch))
            self.current_batch = None
            return True
        return False

class ModelRunner(BasePipelineStage):
    def __init__(self, model, task_queue, result_queue):
        super().__init__()
        self.model = model.cuda().eval()
        self.task_queue = task_queue
        self.result_queue = result_queue
        self.stream = torch.cuda.Stream()
        self.current_batch = None
        self.counter = 0

    def trigger_process(self):
        if not self._stop_event.is_set() and self.result_queue.qsize() < 3:
            try:
                item = self.task_queue.get_nowait()
                if item[0] == 'data':
                    with torch.cuda.stream(self.stream):
                        self.current_batch = self.model(item[1])
                    return True
                elif item[0] == 'end':
                    self._stop_event.set()
                    self.result_queue.put(('end', None))
                    return False
            except queue.Empty:
                pass
        return False

    def complete_process(self):
        if self.current_batch is not None:
            torch.cuda.current_stream().wait_stream(self.stream)
            self.result_queue.put(('result', self.current_batch))
            self.current_batch = None
            return True
        return False

class ResultProcessor(BasePipelineStage):
    def __init__(self, result_queue):
        super().__init__()
        self.result_queue = result_queue
        self.stream = torch.cuda.Stream()
        self.processed_count = 0

    def trigger_process(self):
        if not self._stop_event.is_set():
            try:
                item = self.result_queue.get_nowait()
                if item[0] == 'result':
                    with torch.cuda.stream(self.stream):
                        self.processed_count += 1
                    return True
                elif item[0] == 'end':
                    self._stop_event.set()
                    return False
            except queue.Empty:
                pass
        return False

class InferenceScheduler:
    def __init__(self, model, dataset, batch_size=32):
        self.task_queue = queue.Queue(maxsize=3)
        self.result_queue = queue.Queue(maxsize=3)
        self.components = {
            'loader': DataLoader(dataset, self.task_queue, batch_size),
            'model': ModelRunner(model, self.task_queue, self.result_queue),
            'processor': ResultProcessor(self.result_queue)
        }
        self.active = True

    def random_trigger(self):
        """Randomly trigger a pipeline stage if conditions are met"""
        stage = random.choice(list(self.components.values()))
        
        if isinstance(stage, DataLoader):
            if stage.trigger_load():
                stage.complete_load()
        elif isinstance(stage, ModelRunner):
            if stage.trigger_process():
                stage.complete_process()
        elif isinstance(stage, ResultProcessor):
            stage.trigger_process()

    def run(self):
        """Main execution loop with event-driven scheduling"""
        while self.active:
            self.random_trigger()
            # Check completion conditions
            if (self.components['loader']._stop_event.is_set() and
                self.components['model']._stop_event.is_set() and
                self.components['processor']._stop_event.is_set()):
                self.active = False

if __name__ == "__main__":
    model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
    dataset = InferenceDataset(size=1000)
    
    scheduler = InferenceScheduler(model, dataset, batch_size=32)
    scheduler.run()