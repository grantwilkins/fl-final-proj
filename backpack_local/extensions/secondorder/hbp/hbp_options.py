class LossHessianStrategy:
    EXACT = "exact"
    SAMPLING = "sampling"
    SUM = "sum"

    CHOICES = [
        EXACT,
        SAMPLING,
        SUM,
    ]

    @classmethod
    def check_exists(cls, strategy):
        if strategy not in cls.CHOICES:
            raise AttributeError(
                f"Unknown loss Hessian strategy: {strategy}. "
                + f"Expecting one of {cls.CHOICES}"
            )


class BackpropStrategy:
    SQRT = "sqrt"
    BATCH_AVERAGE = "average"

    CHOICES = [
        BATCH_AVERAGE,
        SQRT,
    ]

    @classmethod
    def is_batch_average(cls, strategy):
        cls.check_exists(strategy)
        return strategy == cls.BATCH_AVERAGE

    @classmethod
    def is_sqrt(cls, strategy):
        cls.check_exists(strategy)
        return strategy == cls.SQRT

    @classmethod
    def check_exists(cls, strategy):
        if strategy not in cls.CHOICES:
            raise AttributeError(
                f"Unknown backpropagation strategy: {strategy}. "
                + f"Expect {cls.CHOICES}"
            )


class ExpectationApproximation:
    BOTEV_MARTENS = "E[J^T E(H) J]"
    CHEN = "E(J^T) E(H) E(J)"

    CHOICES = [
        BOTEV_MARTENS,
        CHEN,
    ]

    @classmethod
    def should_average_param_jac(cls, strategy):
        cls.check_exists(strategy)
        return strategy == cls.CHEN

    @classmethod
    def check_exists(cls, strategy):
        if strategy not in cls.CHOICES:
            raise AttributeError(
                f"Unknown EA strategy: {strategy}. " + f"Expect {cls.CHOICES}"
            )
