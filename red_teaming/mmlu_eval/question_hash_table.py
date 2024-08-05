import torch
import numpy as np

# This data structure stores both the correctness and the array of A, B, C, D probabilities for each question.
# This data structure is used to ensure that prompting the model with the same question multiple times has no effect on the overall accuracy.
# It is also helpful for eventually writing this metadata to a file.

class QuestionHashTable:
    def __init__(self):
        self.hash_table = {}
        self.serialize_table = []

    def add_entry(self, tensor):
        tensor_hash = hash(tuple(tensor[:(tensor.shape[-1] - 5)].view(-1).tolist()))
        if tensor_hash not in self.hash_table:
            self.hash_table[tensor_hash] = tensor
            self.serialize_table.append(tensor)
        # We will assume that hash collisions are extremely unlikely

    def get_serialized_table(self):
        return self.serialize_table
    
    def compute_accuracy(self):
        correct = 0
        total = 0
        for tensor in self.serialize_table:
            if tensor[-1].item():
                correct += 1
            total += 1
        return np.float32(correct) / total


def update_serialized_table(accelerator, serialized_table, iid_batch, probs_batch, labels_batch, preds_batch, truthy_array):
    # The serialized table stores the following information for each question:
    # - The input IDs
    # - The probabilities of each answer choice
    # - The correct answer
    # - The model's prediction
    # - The correctness of the model's prediction
    # This information is stored in a single tensor for each question, and is used to output both the final accuracies and the question-level metadata.

    iid_tensors = torch.chunk(iid_batch, iid_batch.shape[0], dim=0)
    probs_tensors = torch.chunk(probs_batch, probs_batch.shape[0], dim=0)
    truthy_array = torch.tensor(truthy_array, dtype=torch.bool)

    labels_tensor = torch.tensor([["A", "B", "C", "D"].index(label) for label in labels_batch], dtype=torch.long)
    preds_tensor = torch.tensor([["A", "B", "C", "D"].index(pred) for pred in preds_batch], dtype=torch.long)

    for iid, probs, label, pred, truthy in zip(iid_tensors, probs_tensors, labels_tensor, preds_tensor, truthy_array):
        iid = iid.to(accelerator.device)
        probs = probs.to(accelerator.device)
        label = torch.reshape(label, (1, 1)).to(accelerator.device)
        pred = torch.reshape(pred, (1, 1)).to(accelerator.device)
        truthy = torch.reshape(truthy, (1, 1)).to(accelerator.device)
    
        concatenated_tensor = torch.cat((iid, probs, label, pred, truthy), dim=1)
        serialized_table.append(concatenated_tensor)
    return serialized_table