import re
import keras
import pandas as pd

REGEX = r"(\S+):\s(\d+\.\d+)"
PATTERN = re.compile(REGEX)

def intercept_keras_metrics(line):
    for match in PATTERN.finditer(line):
        name = match.group(1)
        value = match.group(2)
        


class MetricsToCSVCallback(keras.callbacks.Callback):
    def __init__(self, filepath, save_freq=100):
        import pandas as pd
        super().__init__()
        self.filepath = filepath
        self.save_freq = save_freq
        self.data = []
        
    def on_train_begin(self, logs=None):
        self.epochs = self.params['epochs']
        self.steps = self.params['steps']
        
    def on_epoch_begin(self, epoch, logs=None):
        self.current_epoch = epoch

    def on_train_batch_end(self, batch, logs=None):
        logs = logs or {}
        step = self.current_epoch * self.steps + batch
        
        # Capture all available metrics
        metrics = {name: logs.get(name, 0) for name in self.model.metrics_names}
        metrics.update({'epoch': self.current_epoch, 'step': step})
        
        self.data.append(metrics)
        
        # Save to CSV periodically
        if step % self.save_freq == 0:
            self.save_to_csv()

    def on_epoch_end(self, epoch, logs=None):
        self.save_to_csv()

    def save_to_csv(self):
        df = pd.DataFrame(self.data)
        df.to_csv(self.filepath, index=False)
        print(f"Metrics saved to {self.filepath}")

    def on_train_end(self, logs=None):
        self.save_to_csv()