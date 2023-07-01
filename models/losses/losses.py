import torch
from torch.nn.functional import triplet_margin_loss, binary_cross_entropy_with_logits
from typing import final, Optional

DEFAULT_GAMMA: final = 0.8
REDUCTIONS: final = frozenset(["mean", "sum", "none"])


def gamma_quadruplet_loss(x_anchor: torch.Tensor,
                          x_pos: torch.Tensor,
                          x_part: torch.Tensor,
                          x_neg: torch.Tensor,
                          gamma: float = DEFAULT_GAMMA,
                          margin_pos_neg: float = 1.0,
                          margin_pos_part: float = 1.0,
                          margin_part_neg: float = 1.0,
                          p: float = 2.0,
                          swap: bool = False,
                          reduction: str = "mean") -> torch.Tensor:
    if gamma < 0 or gamma > 1:
        raise ValueError(f"gamma must be between 0 and 1, {gamma} given")
    if margin_pos_neg <= 0:
        raise ValueError(f"margin_pos_neg must be positive, {margin_pos_neg} given")
    if margin_pos_part <= 0:
        raise ValueError(f"margin_pos_part must be positive, {margin_pos_part} given")
    if margin_part_neg <= 0:
        raise ValueError(f"margin_part_neg must be positive, {margin_part_neg} given")
    if reduction not in REDUCTIONS:
        raise ValueError(f"reduction must be one of: {REDUCTIONS}, "
                         f"{reduction} given")
    if p <= 0:
        raise ValueError(f"p must be positive, {p} given")

    # Compute the triplet losses with no reduction, shape (B,)
    a = triplet_margin_loss(
        anchor=x_anchor,
        positive=x_pos,
        negative=x_neg,
        margin=margin_pos_neg,
        p=p,
        swap=swap,
        reduction='none'
    )
    b = triplet_margin_loss(
        anchor=x_anchor,
        positive=x_part,
        negative=x_neg,
        margin=margin_part_neg,
        p=p,
        swap=swap,
        reduction='none'
    )
    c = triplet_margin_loss(
        anchor=x_anchor,
        positive=x_pos,
        negative=x_part,
        margin=margin_pos_part,
        p=p,
        swap=swap,
        reduction='none'
    )

    # Return the reduced loss if required
    if reduction == 'none':
        return a + gamma * b + (1 - gamma) * c
    elif reduction == 'sum':
        return a.sum() + (gamma * b).sum() + ((1 - gamma) * c).sum()
    else:
        return a.mean() + (gamma * b).mean() + ((1 - gamma) * c).mean()


def d_regularized_quadruplet_loss(
        x_anchor: torch.Tensor,
        x_pos: torch.Tensor,
        x_part: torch.Tensor,
        x_neg: torch.Tensor,
        margin_pos_neg: float = 1.0,
        margin_part_neg: float = 1.0,
        lmbd: float = 0.1,
        discr: Optional[torch.nn.Module] = None,
        discr_logits_pos: Optional[torch.Tensor] = None,
        discr_logits_part: Optional[torch.Tensor] = None,
        p: float = 2.0,
        swap: bool = False,
        reduction: str = "mean") -> torch.Tensor:
    if lmbd <= 0:
        raise ValueError(f"lmbd must be positive, {lmbd} given")
    if margin_pos_neg <= 0:
        raise ValueError(f"margin_pos_neg must be positive, {margin_pos_neg} given")
    if margin_part_neg <= 0:
        raise ValueError(f"margin_part_neg must be positive, {margin_part_neg} given")
    if reduction not in REDUCTIONS:
        raise ValueError(f"reduction must be one of: {REDUCTIONS}, "
                         f"{reduction} given")
    if p <= 0:
        raise ValueError(f"p must be positive, {p} given")
    if discr is None and (discr_logits_part is None or discr_logits_pos is None):
        raise ValueError(f"Either discriminator or discriminator logits must be "
                         f"given")

    # Compute the triplet losses with no reduction, shape (B,)
    a = triplet_margin_loss(
        anchor=x_anchor,
        positive=x_pos,
        negative=x_neg,
        margin=margin_pos_neg,
        p=p,
        swap=swap,
        reduction='none'
    )
    b = triplet_margin_loss(
        anchor=x_anchor,
        positive=x_part,
        negative=x_neg,
        margin=margin_part_neg,
        p=p,
        swap=swap,
        reduction='none'
    )

    # Compute logits if required, with shape (B, 1)
    if discr_logits_pos is None or discr_logits_part is None:
        discr_logits_pos = discr(x_anchor, x_pos)
        discr_logits_part = discr(x_anchor, x_part)

    # Unsqueeze logits to obtain tensors with shape (B, 1, 1)
    discr_logits_pos = discr_logits_pos.unsqueeze(1)
    discr_logits_part = discr_logits_part.unsqueeze(1)

    # Concatenate the logits and create targets with the same shape (B, 2, 1)
    discr_logits_cat = torch.cat([discr_logits_pos, discr_logits_part], dim=1)
    target_pos = torch.ones_like(discr_logits_pos)
    target_part = torch.zeros_like(discr_logits_part)
    target_cat = torch.cat([target_pos, target_part], dim=1)

    # Calculate BCE loss with no reduction, shape (B, 2, 1)
    bce = binary_cross_entropy_with_logits(discr_logits_cat, target=target_cat,
                                           reduction='none')

    # Sum loss value over the 2-th dim, obtaining tensor with shape (B, 1, 1)
    bce = bce.sum(dim=1, keepdim=True)

    print(bce.shape)

    # Return the reduced loss if required
    if reduction == 'none':
        return a + b - lmbd * bce.squeeze(dim=-1)
    elif reduction == 'sum':
        return a.sum() + b.sum() - lmbd * bce.squeeze(dim=-1).sum()
    else:
        return a.mean() + b.mean() - lmbd * bce.squeeze(dim=-1).mean()


from abc import ABC, abstractmethod


class QuadrupletLoss(torch.nn.Module, ABC):
    def __init__(self,
                 margin_pos_neg: float = 1.0,
                 margin_pos_part: float = 1.0,
                 p: float = 2.0,
                 swap: bool = False,
                 reduction: str = "mean"):
        super().__init__()
        if margin_pos_neg <= 0:
            raise ValueError(f"margin_pos_neg must be positive, {margin_pos_neg} given")
        if margin_pos_part <= 0:
            raise ValueError(f"margin_pos_part must be positive, {margin_pos_part} given")
        if reduction not in REDUCTIONS:
            raise ValueError(f"reduction must be one of: {REDUCTIONS}, "
                             f"{reduction} given")
        if p <= 0:
            raise ValueError(f"p must be positive, {p} given")

        self.__margin_pos_neg: float = margin_pos_neg
        self.__margin_pos_part: float = margin_pos_part
        self.__p: float = p
        self.__swap: bool = swap
        self.__reduction: str = reduction

    @property
    def margin_pos_neg(self) -> float:
        return self.__margin_pos_neg

    @margin_pos_neg.setter
    def margin_pos_neg(self, margin_pos_neg: float):
        if margin_pos_neg <= 0:
            raise ValueError(f"margin_pos_neg must be positive, {margin_pos_neg} given")
        self.__margin_pos_neg = margin_pos_neg

    @property
    def margin_pos_part(self) -> float:
        return self.__margin_pos_part

    @margin_pos_part.setter
    def margin_pos_part(self, margin_pos_part: float):
        if margin_pos_part <= 0:
            raise ValueError(f"margin_pos_part must be positive, {margin_pos_part} given")
        self.__margin_pos_part = margin_pos_part

    @property
    def p(self) -> float:
        return self.__p

    @p.setter
    def p(self, p: float):
        if p <= 0:
            raise ValueError(f"p must be positive, {p} given")
        self.__p = p

    @property
    def swap(self) -> bool:
        return self.__swap

    @swap.setter
    def swap(self, swap: bool):
        self.__swap = swap

    @property
    def reduction(self) -> str:
        return self.__reduction

    @reduction.setter
    def reduction(self, reduction: str):
        if reduction not in REDUCTIONS:
            raise ValueError(f"reduction must be one of: {REDUCTIONS}, "
                             f"{reduction} given")
        self.__reduction = reduction

    @abstractmethod
    def forward(self,
                x_anchor: torch.Tensor,
                x_pos: torch.Tensor,
                x_part: torch.Tensor,
                x_neg: torch.Tensor,
                reduction: Optional[str] = None,
                **kwargs) -> torch.Tensor:
        raise NotImplementedError()


class GammaQuadrupletLoss(QuadrupletLoss):
    def __init__(self,
                 gamma: float = DEFAULT_GAMMA,
                 margin_pos_neg: float = 1.0,
                 margin_pos_part: float = 1.0,
                 margin_part_neg: float = 1.0,
                 p: float = 2.0,
                 swap: bool = False,
                 reduction: str = "mean"):
        super().__init__(margin_pos_part=margin_pos_part,
                         margin_pos_neg=margin_pos_neg,
                         p=p,
                         swap=swap,
                         reduction=reduction)
        if gamma < 0 or gamma > 1:
            raise ValueError(f"gamma must be between 0 and 1, {gamma} given")
        if margin_part_neg <= 0:
            raise ValueError(f"margin_part_neg must be positive, {margin_part_neg} given")

        self.__gamma: float = gamma
        self.__margin_part_neg: float = margin_part_neg

    @property
    def gamma(self) -> float:
        return self.__gamma

    @gamma.setter
    def gamma(self, gamma: float):
        if gamma < 0 or gamma > 1:
            raise ValueError(f"gamma must be between 0 and 1, {gamma} given")
        self.__gamma = gamma

    @property
    def margin_part_neg(self) -> float:
        return self.__margin_part_neg

    @margin_part_neg.setter
    def margin_part_neg(self, margin_part_neg: float):
        if margin_part_neg <= 0:
            raise ValueError(f"margin_part_neg must be positive, {margin_part_neg} given")
        self.__margin_part_neg = margin_part_neg

    def forward(self,
                x_anchor: torch.Tensor,
                x_pos: torch.Tensor,
                x_part: torch.Tensor,
                x_neg: torch.Tensor,
                reduction: Optional[str] = None,
                **kwargs) -> torch.Tensor:

        reduction = self.reduction if reduction is None else reduction

        return gamma_quadruplet_loss(x_anchor=x_anchor,
                                     x_pos=x_pos,
                                     x_part=x_part,
                                     x_neg=x_neg,
                                     gamma=self.gamma,
                                     margin_pos_neg=self.margin_pos_neg,
                                     margin_pos_part=self.margin_pos_part,
                                     margin_part_neg=self.margin_part_neg,
                                     p=self.p,
                                     swap=self.swap,
                                     reduction=reduction)

