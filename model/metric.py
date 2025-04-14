import torch

def accuracy(output, target):
  with torch.no_grad():
    pred = torch.argmax(output, dim=1)
    assert pred.shape[0] == len(target)
    correct = 0
    correct += torch.sum(pred == target).item()
  return correct / len(target)


def top_k_acc(output, target, k=5):
  with torch.no_grad():
    pred = torch.topk(output, k, dim=1)[1]
    assert pred.shape[0] == len(target)
    correct = 0
    for i in range(k):
      correct += torch.sum(pred[:, i] == target).item()
  return correct / len(target)

def multiclass_confusion_matrix(input: torch.Tensor,
    target: torch.Tensor,
    num_classes = 60) -> torch.Tensor:
  confusion_matrix = torch.zeros(num_classes, num_classes)
  with torch.no_grad():
    for t, p in zip(target.view(-1), input.view(-1)):
      confusion_matrix[t.long(), p.long()] += 1
  return confusion_matrix
