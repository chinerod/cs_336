# Logging Utils

```python
import logging
import sys
from pathlib import Path
from datetime import datetime


def setup_logger(
    name: str = "cs336",
    log_file: str = None,
    level: int = logging.INFO,
    console: bool = True
):
    """
    设置日志记录器

    Args:
        name: 日志器名称
        log_file: 日志文件路径
        level: 日志级别
        console: 是否输出到控制台

    Returns:
        logger: 配置好的日志器
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # 清除已有处理器
    logger.handlers.clear()

    # 格式化
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # 文件输出
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setLevel(level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    # 控制台输出
    if console:
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(level)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    return logger


class TrainingLogger:
    """训练日志记录器"""

    def __init__(self, log_dir: str = "./logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.log_dir / f"train_{timestamp}.log"

        self.logger = setup_logger(
            name="training",
            log_file=str(self.log_file),
            console=True
        )

        self.step = 0
        self.epoch = 0

    def log_step(self, loss: float, lr: float = None, **kwargs):
        """记录训练步骤"""
        msg = f"Epoch {self.epoch}, Step {self.step}, Loss: {loss:.4f}"
        if lr:
            msg += f", LR: {lr:.6f}"
        for k, v in kwargs.items():
            msg += f", {k}: {v:.4f}"
        self.logger.info(msg)
        self.step += 1

    def log_epoch(self, epoch: int, train_loss: float, val_loss: float = None):
        """记录epoch"""
        self.epoch = epoch
        msg = f"Epoch {epoch} - Train Loss: {train_loss:.4f}"
        if val_loss:
            msg += f", Val Loss: {val_loss:.4f}"
        self.logger.info(msg)

    def log_metrics(self, metrics: dict, step: int = None):
        """记录指标"""
        step = step or self.step
        msg = f"Step {step}: " + ", ".join([f"{k}={v:.4f}" for k, v in metrics.items()])
        self.logger.info(msg)
```

---

*日志工具*
