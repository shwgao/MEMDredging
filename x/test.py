from pipeline import PipelineManager

if __name__ == "__main__":
    # Simulated input data source and model
    input_data = ["image1", "image2", "image3"]
    model = "DummyModel"  # Replace with your actual model object

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