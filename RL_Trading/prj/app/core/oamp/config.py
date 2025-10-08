class ConfigOAMP:
    def __init__(
        self,
        args: dict,
    ):
        self.agents_weights_upd_freq = args.get("agents_weights_upd_freq", 10)
        self.agents_sample_freq = args.get("agents_sample_freq", 10)
        self.loss_fn_window = args.get("loss_fn_window", 30)
        self.action_aggregation_type = args.get("action_aggregation_type", "threshold")
        self.action_threshold = args.get("action_threshold", 0.2)
        self.lr_ub = args.get("lr_ub", 1 / 4)

