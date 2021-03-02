import numpy as np
import time, datetime
import matplotlib.pyplot as plt
import os

class MetricLogger:
    def __init__(self, save_dir):
        self.save_log = os.path.join(save_dir, "log")
        with open(self.save_log, "w") as f:
            f.write(
                f"{'Episode':>8}{'Step':>8}{'Epsilon':>10}{'MeanReward':>15}"
                f"{'MeanLength':>15}{'MeanLoss':>15}{'MeanQValue':>15}"
                f"{'TimeDelta':>15}{'Time':>20}\n"
            )
