import torch
import numpy as np

class QuestionHashTable:
    """
    A data structure to store question-related information, ensuring correctness and allowing for serialization.
    This is used to prevent duplicate questions from affecting accuracy calculations and to facilitate metadata output.
    """

    def __init__(self):
        self.hash_table = {}  # Dictionary to store unique questions
        self.serialize_table = []  # List to maintain order for serialization

    def add_entry(self, tensor):
        """
        Add a new entry to the hash table if it doesn't already exist.
        
        Args:
        tensor (torch.Tensor): Tensor containing question data.
        """
        # Create a hash of the tensor (excluding the last 5 elements)
        tensor_hash = hash(tuple(tensor[: (tensor.shape[-1] - 5)].view(-1).tolist()))
        if tensor_hash not in self.hash_table:
            self.hash_table[tensor_hash] = tensor
            self.serialize_table.append(tensor)
        # Note: Assumes hash collisions are extremely unlikely

    def get_serialized_table(self):
        """Return the serialized table of entries."""
        return self.serialize_table

    def compute_accuracy(self):
        """
        Compute the accuracy based on the stored entries.
        
        Returns:
        float: The computed accuracy.
        """
        correct = 0
        total = 0
        for tensor in self.serialize_table:
            if tensor[-1].item():  # Check if the last element (correctness) is True
                correct += 1
            total += 1
        return np.float32(correct) / total

def update_serialized_table(
    accelerator,
    serialized_table,
    iid_batch,
    probs_batch,
    labels_batch,
    preds_batch,
    truthy_array,
):
    """
    Update the serialized table with new batch information.
    
    Args:
    accelerator: The accelerator being used (likely for device management).
    serialized_table (list): The table to update.
    iid_batch (torch.Tensor): Batch of input IDs.
    probs_batch (torch.Tensor): Batch of probabilities for each answer choice.
    labels_batch (list): Batch of correct labels.
    preds_batch (list): Batch of model predictions.
    truthy_array (list): Batch of boolean values indicating correctness.
    
    Returns:
    list: The updated serialized table.
    """
    # Convert label characters to indices
    labels_tensor = torch.tensor(
        [["A", "B", "C", "D"].index(label) for label in labels_batch], dtype=torch.long
    )
    preds_tensor = torch.tensor(
        [["A", "B", "C", "D"].index(pred) for pred in preds_batch], dtype=torch.long
    )
    
    # Convert truthy_array to a boolean tensor
    truthy_array = torch.tensor(truthy_array, dtype=torch.bool)
    
    # Process each item in the batch
    for iid, probs, label, pred, truthy in zip(
        torch.chunk(iid_batch, iid_batch.shape[0], dim=0),  # Split input IDs into individual tensors
        torch.chunk(probs_batch, probs_batch.shape[0], dim=0),  # Split probabilities into individual tensors
        labels_tensor,
        preds_tensor,
        truthy_array
    ):
        # Move tensors to the appropriate device
        iid = iid.to(accelerator.device)
        probs = probs.to(accelerator.device)
        label = torch.reshape(label, (1, 1)).to(accelerator.device)
        pred = torch.reshape(pred, (1, 1)).to(accelerator.device)
        truthy = torch.reshape(truthy, (1, 1)).to(accelerator.device)
        
        # Concatenate all information into a single tensor
        concatenated_tensor = torch.cat((iid, probs, label, pred, truthy), dim=1)
        serialized_table.append(concatenated_tensor)
    
    return serialized_table