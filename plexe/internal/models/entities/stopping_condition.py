import time

from plexe.internal.models.entities.metric import Metric


class StoppingCondition:
    """
    A class to represent a stopping condition for an optimization process.
    """

    def __init__(self, max_generations: int = None, max_time: int = None, metric: Metric = None):
        """
        Initialize the StoppingCondition with the given parameters.

        :param max_generations: max number of solutions to try before giving up
        :param max_time: max time to spend on optimization, in seconds
        :param metric: threshold for the optimization metric, stop once this is reached
        """
        if not any([max_generations, max_time, metric]):
            raise ValueError("At least one stopping condition must be provided")

        self.max_generations = max_generations
        self.max_time = max_time
        self.metric = metric

    def is_met(self, generations: int, start_time: float, metric: Metric) -> bool:
        return (
            self.max_generations
            and generations >= self.max_generations
            or self.max_time
            and time.time() - start_time >= self.max_time
            or self.metric
            and metric >= self.metric
        )

    def __repr__(self) -> str:
        """
        Return a string representation of the Metric object.

        :return: A string representation of the Metric.
        """
        return (
            f"StoppingCondition(max_nodes={self.max_generations!r}, max_time={self.max_time}, metric={self.metric!r})"
        )

    def __str__(self) -> str:
        """
        Return a user-friendly string representation of the Metric.

        :return: A string describing the Metric.
        """
        msg = "stop after at least one condition is met: "
        if self.max_generations:
            msg += f"attempted {self.max_generations} solutions | "
        if self.max_time:
            msg += f"elapsed {self.max_time} seconds | "
        if self.metric:
            msg += f"reached performance of at least '{self.metric}'"

        return msg
