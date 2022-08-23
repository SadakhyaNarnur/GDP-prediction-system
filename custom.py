from sklearn.ensemble._forest import ForestRegressor
from sklearn.neural_network import MLPRegressor

class WeightedMLPRegressor(MLPRegressor):
    def fit(self, X, y, **kwargs):
        super().fit(X, y)
    
    def predict(self, X, **kwargs):
        return super().predict(X)

# Creating our Model class and initializing all  parameters
class RFMLPRegressor(ForestRegressor):
    def __init__(
        self,
        n_estimators=100,
        *,
        criterion="squared_error",
        hidden_layer_sizes=(100,),       
        activation="relu",
        solver="adam",
        alpha=0.0001,
        batch_size="auto",
        learning_rate="constant",
        learning_rate_init=0.001,
        power_t=0.5,
        max_iter=200,
        shuffle=True,
        random_state=None,
        tol=1e-4,
        verbose=False,
        warm_start=False,
        momentum=0.9,
        nesterovs_momentum=True,
        early_stopping=False,
        validation_fraction=0.1,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-8,
        n_iter_no_change=10,
        max_fun=15000,

        bootstrap=True,
        oob_score=False,
        n_jobs=None,
        max_samples=None
    ):
        # Taking MLP as base estimator in place of Decision tree
        super().__init__(
            base_estimator=WeightedMLPRegressor(),
            n_estimators=n_estimators,
            estimator_params=(
            "hidden_layer_sizes",
            "activation",
            "solver",
            "alpha",
            "batch_size",
            "learning_rate",
            "learning_rate_init",
            "power_t",
            "max_iter",
            "shuffle",
            "random_state",
            "tol",
            "warm_start",
            "momentum",
            "nesterovs_momentum",
            "early_stopping",
            "validation_fraction",
            "beta_1",
            "beta_2",
            "epsilon",
            "n_iter_no_change",
            "max_fun",
            ),
            bootstrap=bootstrap,
            oob_score=oob_score,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            warm_start=warm_start,
            max_samples=max_samples,
        )
        self.criterion = criterion
        self.hidden_layer_sizes=hidden_layer_sizes
        self.activation=activation
        self.solver=solver
        self.alpha=alpha
        self.batch_size=batch_size
        self.learning_rate=learning_rate
        self.learning_rate_init=learning_rate_init
        self.power_t=power_t
        self.max_iter=max_iter
        self.shuffle=shuffle
        self.tol=tol
        self.momentum=momentum
        self.nesterovs_momentum=nesterovs_momentum
        self.early_stopping=early_stopping
        self.validation_fraction=validation_fraction
        self.beta_1=beta_1
        self.beta_2=beta_2
        self.epsilon=epsilon
        self.n_iter_no_change=n_iter_no_change
        self.max_fun=max_fun

