from abc import ABC, abstractmethod
import logging
import torch


LOGGER = logging.getLogger(__name__)


class Callback(ABC):
    def __init__(self):
        self.__epoch = -1
        self.__steps = -1

    @property
    def epoch(self) -> int:
        return self.__epoch

    @epoch.setter
    def epoch(self, epoch: int):
        if self.__epoch > epoch >= 0:
            raise ValueError(
                f"The new epoch # must be greater or equal than the previous one ({self.__epoch}), {epoch} given."
            )
        self.__epoch = epoch

    @property
    def steps(self) -> int:
        return self.__steps

    @steps.setter
    def steps(self, steps: int):
        if self.__steps > steps >= -1:
            raise ValueError(
                f"The new # of steps must be greater or equal than the previous one ({self.__steps}), {steps} given."
            )
        self.__steps = steps

    def update_epoch_steps(self, epoch: int, steps: int):
        self.epoch = epoch
        self.steps = steps

    @abstractmethod
    def __call__(self, score: float, epoch: int, steps: int, *args, **kwargs):
        raise NotImplementedError(f"Every {self.__class__} subclass should override __call__ method.")


class EarlyStoppingException(BaseException):
    def __init__(self, epoch, best_epoch, best_score, counter):
        super().__init__(
            f"EarlyStoppingException: score did not improve for {counter} epochs, "
            f"early stopping at epoch {epoch}, restoring best model at epoch: "
            f"{best_epoch} with score: {best_score}."
        )


class EarlyStoppingCallback(Callback):
    def __init__(self, patience: int = 10, delta: float = 0.0, minimization: bool = True):
        super().__init__()
        self._patience = patience
        self._delta = delta
        self._minimization = minimization
        self._counter: int = 0
        self._best_score = torch.inf if minimization else -torch.inf
        self._best_epoch = -1

    def __call__(self, score: float, epoch: int, steps: int, *args, **kwargs):
        self.update_epoch_steps(epoch, steps)

        diff = self._best_score - score if self._minimization else score - self._best_score

        if diff > 0:
            self._best_score = score
            self._best_epoch = epoch

        if diff > self._delta:
            self._counter = 0
        else:
            self._counter += 1
            info = f"Loss didn't improve from last epoch. Early stopping counter increased to: " \
                   f"{self._counter}/{self._patience}."
            LOGGER.info(info)
            print(info)

        if self._counter >= self._patience:
            raise EarlyStoppingException(
                epoch=epoch, best_epoch=self._best_epoch, best_score=self._best_score, counter=self._counter
            )
