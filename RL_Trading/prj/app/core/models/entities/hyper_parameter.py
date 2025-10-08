import optuna
import optuna.distributions as distributions
import typing

from munch import DefaultMunch


class HyperParameter:

    def __init__(self, config: DefaultMunch):
        #self.family = config.family
        self.name = config.name
        if config.type == 'int':
            self.distribution = distributions.IntDistribution(config.min, config.max, step=config.step)
        elif config.type == 'log':
            self.distribution = distributions.FloatDistribution(config.min, config.max, log=True)
        elif config.type == 'float':
            self.distribution = distributions.FloatDistribution(config.min, config.max, step=config.step)
        elif config.type == 'cat':
            self.distribution = distributions.CategoricalDistribution(config.choices)
        else:
            raise AttributeError(f"Invalid distribution_type {self.distribution_type}.")

    def suggest_value(self, trial: optuna.Trial) -> typing.Union[int, float]:
        # Parameter is fixed
        if self.is_fixed():
            return self.distribution.low
        # Parameter has a distribution
        if type(self.distribution) == distributions.CategoricalDistribution:
            return trial.suggest_categorical(self.name, self.distribution.choices)
        elif type(self.distribution) == distributions.IntDistribution:
            return trial.suggest_int(
                self.name,
                self.distribution.low,
                self.distribution.high,
                step=self.distribution.step
            )
        else:
            return trial.suggest_float(
                self.name,
                self.distribution.low,
                self.distribution.high,
                log=self.distribution.log,
                step=self.distribution.step
            )

    def is_fixed(self) -> bool:
        return self.distribution.single()
