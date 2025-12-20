from src.train import Trainer

class ScheduledSamplingTrainer(Trainer):
    def __init__(
        self,
        *args,
        tf_start=1.0,
        tf_end=0.2,
        tf_decay_steps=50000,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.tf_start = tf_start
        self.tf_end = tf_end
        self.tf_decay_steps = tf_decay_steps
    
    def teacher_forcing_ratio(self):
        if self.global_step >= self.tf_decay_steps:
            return self.tf_end
        return self.tf_start - (self.tf_start - self.tf_end) * (self.global_step / self.tf_decay_steps)