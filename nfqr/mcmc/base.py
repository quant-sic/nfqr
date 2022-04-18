class MCMC(object):
    def __init__(self, n_steps) -> None:
        self.n_steps = n_steps
        self._config = None
        self.n_accepted = 0
        self.n_current_steps = 0

    def step(self):
        pass

    def initialize(self):
        pass

    @property
    def current_config(self):
        return self._config

    @current_config.setter
    def current_config(self, new_current_config):
        self._config = new_current_config

    def __iter__(self):

        self.initialize()

        for _ in range(self.n_steps):
            self.n_current_steps += 1
            self.step()
            yield self.current_config

    def run_entire_chain(self):
        for _ in self:
            pass

    @property
    def acceptance_ratio(self):
        return self.n_accepted / self.n_current_steps
