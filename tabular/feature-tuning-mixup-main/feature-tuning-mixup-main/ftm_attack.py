import torch

def fgsm_attack(model, x, y, epsilon=0.1):
    x.requires_grad = True

    output = model(x)
    loss = torch.nn.functional.cross_entropy(output, y)

    model.zero_grad()
    loss.backward()

    x_adv = x + epsilon * x.grad.sign()
    return x_adv.detach()