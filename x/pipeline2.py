from base import *   
import threading
import torch.multiprocessing as mp
import queue
import torch
import time
from torch.utils.data import Dataset, DataLoader

class InferenceDataset(Dataset):
    def __init__(self, size=1000):
        self.data = torch.randn(size, 3, 224, 224)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

class PipelineContext:
    def __init__(self, task_queue=None, result_queue=None, 
                process_event=None, load_event=None, 
                result_event=None, done_event=None, result_full_event=None, device='cuda'):
        self.task_queue = task_queue
        self.result_queue = result_queue
        self.process_event = process_event
        self.load_event = load_event
        self.result_event = result_event
        self.process_done_event = done_event
        self.result_full_event = result_full_event
        self.device = device
        
        
class BasePipelineStage:
    def __init__(self, context=None):
        self._stop_event = mp.Event()
        self.context = context or PipelineContext()

    def start(self):
        """Start the stage's process"""
        self._run()

    def stop(self):
        """Signal the stage to stop"""
        self._stop_event.set()

    def _run(self):
        """Main execution loop to be implemented by subclasses"""
        raise NotImplementedError


class DataPreLoader(BasePipelineStage):
    def __init__(self, dataset, context, batch_size=32):
        super().__init__(context)
        self.loader = DataLoader(dataset, 
                               batch_size=batch_size,
                               pin_memory=(self.context.device == 'cuda'))
        
    def _run(self):
        # Create stream in the process where it will be used
        stream = torch.cuda.Stream(device=self.context.device)
        counter = 0
        data_iter = iter(self.loader)
        while not self._stop_event.is_set():
            try:
                batch = next(data_iter)
            except StopIteration:
                # if data_iter is exhausted, send stop signal
                self.context.task_queue.put(None)
                print(f"DataLoader: All batches sent {counter}")
                break
            
            if self.context.task_queue.qsize() >= 3:
                self.context.load_event.clear()
                self.context.process_event.set()
                print(f"Passed batch: {batch.shape} (queue full) {counter}")
            
            self.context.load_event.wait()
            
            with torch.cuda.stream(stream):
                batch = batch.to(self.context.device, non_blocking=True)
                print(f"Loaded batch: {batch.shape} on {self.context.device}")
                self.context.task_queue.put(batch)
                
            self.context.process_event.set()
        
        # wait for all batches to be processed
        self.context.process_done_event.wait()
        print("DataLoader exiting cleanly")

    def trigger_load(self, batch):
        pass  # No longer needed since we moved the batch above


class ModelRunner(BasePipelineStage):
    def __init__(self, model, context):
        super().__init__(context)
        self.model = model.to(self.context.device).eval()
        
    def _run(self):
        # Create stream in the process where it will be used
        stream = torch.cuda.Stream(device=self.context.device)
        counter = 0
        with torch.no_grad():
            while not self._stop_event.is_set():
                if self.context.task_queue.empty():
                    self.context.load_event.set()
                    continue
                
                batch = self.context.task_queue.get()
                if batch is None:
                    self.context.result_queue.put(None)
                    break
                
                self.context.process_event.wait()  # Wait for process permission
                
                print(f"ModelRunner event {counter}: Processed batch: {batch.shape}")
                with torch.cuda.stream(stream):
                    result = self.trigger_process(batch)
                    while self.context.result_queue.qsize() >= 30:
                        print(f"ModelRunner event {counter}: Result queue full, waiting for result to be processed")
                        
                    self.context.result_queue.put(result)
                    self.context.process_event.set()
                    self.context.result_event.set()
                
                counter += 1
                
        self.context.process_done_event.set()

    def trigger_process(self, batch):
        return self.model(batch)


class ResultProcessor(BasePipelineStage):
    def __init__(self, context):
        super().__init__(context)
        
    def _run(self):
        # Create stream in the process where it will be used
        stream = torch.cuda.Stream(device=self.context.device)
        counter = 0
        while not self._stop_event.is_set():
            if self.context.result_queue.empty():
                self.context.process_event.set()
                continue
            
            result = self.context.result_queue.get()
            if result is None:
                break
            
            # Process result
            self.context.result_event.wait()
            
            with torch.cuda.stream(stream):
                print(f"ResultProcessor event {counter}: Processed result: {result.shape}")
                self.context.process_event.set()
            counter += 1
            


class InferenceScheduler:
    def __init__(self, model, dataset, batch_size=32):
        # Create shared context
        self.context = PipelineContext(
            task_queue=mp.Queue(maxsize=4),
            result_queue=mp.Queue(maxsize=4),
            process_event=mp.Event(),
            load_event=mp.Event(),
            result_event=mp.Event(),
            done_event=mp.Event(),
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        
        # Initialize components
        self.dataset = dataset
        self.model = model
        self.batch_size = batch_size
        
        # Set initial states
        self.context.load_event.set()
        self.context.process_event.set()

    def run(self):
        # Create components with shared context
        data_loader = DataPreLoader(self.dataset, self.context, self.batch_size)
        model_runner = ModelRunner(self.model, self.context)
        result_processor = ResultProcessor(self.context)

        # Create and start processes
        processes = [
            mp.Process(target=comp.start, daemon=True)
            for comp in [data_loader, model_runner, result_processor]
        ]
        
        # Create CUDA events before starting timing
        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)
        
        torch.cuda.synchronize()  # Ensure clean start
        start_time.record()
        
        for p in processes:
            print(f"Starting process {p.name}")
            p.start()

        # Wait for all processes to finish
        for p in processes:
            p.join()
        end_time.record()
        
        torch.cuda.synchronize()  # Wait for CUDA events to complete
        
        print(f"Inference completed in {start_time.elapsed_time(end_time)} ms")
        
        print("Inference completed!")

if __name__ == "__main__":
    # Set multiprocessing context first
    mp.set_start_method('spawn')
    
    model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
    dataset = InferenceDataset(size=1000)
    
    # Initialize model in main process first
    model = model.to('cuda' if torch.cuda.is_available() else 'cpu')
    
    scheduler = InferenceScheduler(model, dataset, batch_size=320)
    scheduler.run()