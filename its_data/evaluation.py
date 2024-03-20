from collections.abc import Sequence
from typing import Optional, Protocol, TypeVar, overload

from sklearn import metrics

T = TypeVar("T")


class Evaluation(Protocol):
    """
    An evaluation function takes the true and predicted values, and
    optionally labels for the indices of the categories, and returns a
    dictionary mapping each category to its quality metrics
    """

    def __call__(
        self,
        y_true: Sequence[T] | Sequence[Sequence[bool]],
        y_pred: Sequence[T] | Sequence[Sequence[bool]],
        labels: Optional[Sequence[str]],
        **kwargs
    ) -> dict[str, dict[str, float]]:
        ...


@overload
def eval_classification(
    y_true: Sequence[Sequence[bool]],
    y_pred: Sequence[Sequence[bool]],
    labels: Optional[Sequence[str]] = None,
    **kwargs
) -> dict[str, dict[str, float]]:
    ...


@overload
def eval_classification(
    y_true: Sequence[T],
    y_pred: Sequence[T],
    labels: Optional[Sequence[str]] = None,
    **kwargs
) -> dict[str, dict[str, float]]:
    ...


def eval_classification(
    y_true: Sequence[T] | Sequence[Sequence[bool]],
    y_pred: Sequence[T] | Sequence[Sequence[bool]],
    labels: Optional[Sequence[str]] = None,
    **kwargs
) -> dict[str, dict[str, float]]:
    """
    A thin wrapper around sklearn's classification report that both prints the
    human-readible table and returns the report as a dictionary.
    """
    print(metrics.classification_report(y_true, y_pred, labels=labels, **kwargs))

    return metrics.classification_report(
        y_true, y_pred, labels=labels, output_dict=True, **kwargs
    )  # type: ignore
