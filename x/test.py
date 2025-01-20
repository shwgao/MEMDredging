from pipeline import PipelineManager
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import torch
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
print('torch.cuda.is_available():', torch.cuda.is_available())

torch.set_float32_matmul_precision('high')


# args initialization
parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="cosmoflow", help="")
parser.add_argument("--mode", type=str, default="eager", help="eager, multistream")
parser.add_argument("--stream_num", type=int, default=1)
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--batch_num", type=int, default=10)
parser.add_argument("--communication_time", type=bool, default=False)
parser.add_argument("--device", type=str, default="cuda:0")
parser.add_argument("--is_training", type=bool, default=False)
parser.add_argument("--backend", type=str, default="pytorch", help="pytorch, no_caching, cuda")
parser.add_argument("--hardware", type=str, default="V100", help="V100, A100")

args = parser.parse_args()

# model initialization
if args.model == "climax":
    from src.climax import get_model, get_inputs
elif args.model == "enformer":
    from src.enformer import get_model, get_inputs
elif args.model == "climode":
    from src.climode import get_model, get_inputs
elif args.model == "cosmoflow":
    from src.cosmoflow import get_model, get_inputs
else:
    raise ValueError(f"Model {args.model} not supported")

def main():
    # Simulated input data source and model
    input_data = get_inputs(args.batch_size)  ## TODO: get the data loader from each application, maybe we can use fake data for now
    model = get_model(args.model)  # Replace with your actual model object

    # Initialize and run the pipeline
    pipeline = PipelineManager(input_data, model)
    pipeline.start()

    # Simulate input data being added to the pipeline
    for data in input_data:
        pipeline.input_queue.put(data)

    # Signal the end of the data stream
    pipeline.input_queue.put(None)

    # Wait for the pipeline to finish processing
    pipeline.wait_for_completion()
    pipeline.stop()


if __name__ == "__main__":
    main()