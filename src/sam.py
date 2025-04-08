import torch
import monai
from src.sam_modeling import SamModel
from transformers import SamProcessor

seg_loss = monai.losses.DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')

class ModelWrapper(torch.nn.Module):
  # warp the loss computing and model forward pass
  def __init__(self):
    super().__init__()
    self.model = SamModel.from_pretrained("facebook/sam-vit-base")
    self.batch_aggregate = False
    self.mini_batch = 8
    self.checkpointing = False

  def forward(self, input_):
    inputs, ground_truth_masks = input_
    self.model.batch_aggregate = self.batch_aggregate
    self.model.mini_batch = self.mini_batch
    self.model.checkpointing = self.checkpointing
    outputs = self.model(**inputs, multimask_output=False)
    loss = self.compute_loss(outputs.pred_masks, ground_truth_masks)
    return loss
  
  def compute_loss(self, predicted_masks, ground_truth_masks):
    loss = seg_loss(predicted_masks, ground_truth_masks.unsqueeze(2))
    return loss
  
  
def get_model():
    model = ModelWrapper()
    return model

def get_inputs(batch_size):

    processor = SamProcessor.from_pretrained("facebook/sam-vit-base")

    image = torch.randn(batch_size, 3, 1024, 1024)
    image = (image - image.min()) / (image.max() - image.min())
    image = torch.clamp(image, 0, 1)
    # Change prompt to be a list of lists of lists (batch of boxes)
    prompt = [[[0.0, 0.0, 256.0, 256.0]]*10] * batch_size  # Each box is a list of floats, wrapped in a list
    inputs = processor(
      image, 
      input_boxes=prompt,  # Remove the extra list wrapper to match batch size
      return_tensors="pt", 
      do_rescale=False
    )
    
    ground_truth_masks = torch.randn(batch_size, 10, 256, 256)
    
    return (inputs, ground_truth_masks), [0], True

if __name__ == "__main__":
    model = get_model()
    inputs, ground_truth_masks, _ = get_inputs(10)  # Unpack the tuple
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    inputs = {k: v.to(device) for k, v in inputs.items()}  # Move each input tensor to device
    ground_truth_masks = ground_truth_masks.to(device)  # Move masks to device

    with torch.no_grad():
        outputs = model(inputs, ground_truth_masks)  # Pass unpacked arguments
    
    # SAM model returns different outputs, let's inspect them
    print("Model outputs:", outputs)  # First print the full output structure
    if hasattr(outputs, 'iou_scores'):
        print("Shape of iou_scores:", outputs.iou_scores.shape)
    if hasattr(outputs, 'pred_masks'):
        print("Shape of pred_masks:", outputs.pred_masks.shape)
    
    print(torch.cuda.max_memory_allocated() / 1024 ** 3)