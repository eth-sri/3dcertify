import torch
import torch.nn as nn


class Domain:

    def project(self, x: torch.Tensor) -> torch.Tensor:
        pass

    def random_point(self) -> torch.Tensor:
        pass


class EpsBox(Domain):

    def __init__(self, points: torch.Tensor, eps: float):
        super(EpsBox, self).__init__()
        self.points = points.detach()
        self.eps = eps
        self.x_min = self.points - eps
        self.x_max = self.points + eps

    def project(self, x: torch.Tensor) -> torch.Tensor:
        assert self.x_min.size(0) == x.size(0)
        x = torch.max(self.x_min, x)
        x = torch.min(self.x_max, x)
        return x

    def random_point(self) -> torch.Tensor:
        box = torch.empty_like(self.points).uniform_(-self.eps, self.eps)
        return self.points + box


def project_onto_line_segment(point: torch.Tensor, a: torch.Tensor, b: torch.Tensor):
    assert len(point.size()) == 2 and point.size(1) == 3
    assert len(a.size()) == 2 and point.size(1) == 3
    assert len(b.size()) == 2 and point.size(1) == 3

    ab = b - a
    ap = point - a
    factor = batch_dot(ap, ab) / batch_dot(ab, ab)
    factor = torch.clamp(factor, min=0.0, max=1.0)
    return a + factor * ab


def batch_dot(a: torch.Tensor, b: torch.Tensor):
    return torch.sum(a * b, dim=-1)


class FaceBox(Domain):

    def __init__(self, faces: torch.Tensor):
        super(FaceBox, self).__init__()
        self.faces = faces.detach()
        self.x_min = torch.min(self.faces, dim=2)[0]
        self.x_max = torch.max(self.faces, dim=2)[0]

    def project(self, x: torch.Tensor) -> torch.Tensor:
        assert self.faces.size(0) == x.size(0)
        # shape = x.size()
        # x = x.view(-1, 3)
        # faces = self.faces.view(-1, 3, 3)
        # projected = torch.stack([point_triangle_distance(faces[i], x[i]) for i in range(x.size(0))])
        # return projected.view(shape)
        x = torch.max(self.x_min, x)
        x = torch.min(self.x_max, x)
        return x

    def random_point(self) -> torch.Tensor:
        # origin = self.faces[:, :, 0, :]
        # edge1 = self.faces[:, :, 1, :] - origin
        # edge2 = self.faces[:, :, 2, :] - origin
        # a = torch.rand((origin.size(0), origin.size(1), 1), device=origin.device)
        # b = torch.rand((origin.size(0), origin.size(1), 1), device=origin.device)
        # return origin + a * edge1 + b * edge2
        width = (self.x_max - self.x_min)
        offset = torch.empty_like(self.x_min).uniform_() * width
        return self.x_min + offset


def fgsm(model: nn.Module, x: torch.Tensor, target: torch.Tensor, step_size: float):
    input_ = x.clone().detach_()
    input_.requires_grad_()

    logits = model(input_)
    model.zero_grad()
    loss = nn.CrossEntropyLoss()(logits, target)
    loss.backward()

    out = input_ + step_size * input_.grad.sign()
    return out


def pgd(model: nn.Module, domain: Domain, target: torch.Tensor, k: int, step_size: float):
    # initialize with random point from attack domain
    x = domain.random_point()

    for i in range(k):
        # FGSM step
        x = fgsm(model, x, target, step_size)
        # Projection Step
        x = domain.project(x)
    return x
