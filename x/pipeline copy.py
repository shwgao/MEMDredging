from base import *   
import threading
import queue
import torch
from torch.utils.data import Dataset, DataLoader

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

class DataPreLoader(BasePipelineStage):
    def __init__(self, dataset, task_queue, load_event, batch_size=32, device='cuda'):
        super().__init__()
        self.loader = DataLoader(dataset, batch_size=batch_size, pin_memory=True)
        self.task_queue = task_queue
        self.load_event = load_event
        self.device = device

    def _run(self):
        for i, batch in enumerate(self.loader):
            self.load_event.wait()  # Wait for load permission
            print(f"DataLoader event{i}: Loaded batch: {batch.shape}")
            if self._stop_event.is_set():
                break
            self.task_queue.put(batch.to(self.device, non_blocking=True))
        self.task_queue.put(None)  # Termination signal
        
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

class ModelRunner(BasePipelineStage):
    def __init__(self, model, task_queue, result_queue, process_event, device='cuda'):
        super().__init__()
        self.model = model.to(device).eval()
        self.task_queue = task_queue
        self.result_queue = result_queue
        self.process_event = process_event
        self.device = device

    def _run(self):
        counter = 0
        with torch.no_grad():
            while not self._stop_event.is_set():
                batch = self.task_queue.get()
                if batch is None:
                    self.result_queue.put(None)
                    break
                self.process_event.wait()  # Wait for process permission
                print(f"ModelRunner event {counter}: Processed batch: {batch.shape}")
                result = self.model(batch)
                self.result_queue.put(result)
                counter += 1

class ResultProcessor(BasePipelineStage):
    def __init__(self, result_queue, load_event, process_event):
        super().__init__()
        self.result_queue = result_queue
        self.load_event = load_event
        self.process_event = process_event

    def _run(self):
        counter = 0
        while not self._stop_event.is_set():
            result = self.result_queue.get()
            if result is None:
                break
            # Process result
            print(f"ResultProcessor event {counter}: Processed result: {result.shape}")
            counter += 1
            # Control events based on queue size instead of batch size
            if self.result_queue.qsize() >= 2:  # Threshold based on queue capacity
                self.load_event.clear()  # Pause loading when queue is filling up
            else:
                self.load_event.set()  # Resume loading when queue has space
            self.process_event.set()

class InferenceScheduler:
    def __init__(self, model, dataset, batch_size=32):
        self.task_queue = queue.Queue(maxsize=4)
        self.result_queue = queue.Queue(maxsize=4)
        self.load_event = threading.Event()
        self.process_event = threading.Event()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Initialize components
        self.dataset = dataset
        self.model = model
        self.batch_size = batch_size
        
        # Set initial states
        self.load_event.set()
        self.process_event.set()

    def run(self):
        # Create components
        data_loader = DataPreLoader(self.dataset, self.task_queue, self.load_event, self.batch_size, self.device)
        model_runner = ModelRunner(self.model, self.task_queue, self.result_queue, self.process_event, self.device)
        result_processor = ResultProcessor(self.result_queue, self.load_event, self.process_event)

        # Start components and get their threads
        data_loader.start()
        model_runner.start()
        result_processor.start()
        threads = [data_loader.thread, model_runner.thread, result_processor.thread]

        # Monitor and control pipeline
        try:
            # Wait for all tasks to complete
            while True:
                if self.task_queue.empty() and self.result_queue.empty():
                    # Check if all threads have finished
                    if not any(t.is_alive() for t in threads):  # Check thread status
                        break
        except KeyboardInterrupt:
            data_loader.stop()
            model_runner.stop()
            result_processor.stop()
            # Wait for queues to drain
            self.task_queue.join()
            self.result_queue.join()

        print("Inference completed!")

if __name__ == "__main__":
    model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
    dataset = InferenceDataset(size=1000)
    
    scheduler = InferenceScheduler(model, dataset, batch_size=32)
    scheduler.run()