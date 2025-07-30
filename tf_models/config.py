# tf_models/config.py
import pathlib

# Data paths
BASE = pathlib.Path(__file__).parent.parent
DATA_DIR = BASE / "data"

# Filter-width 16, signature 1 & 2
TRAIN16_SIG1 = DATA_DIR / "upd2_n4096l_fw16_PartialTrainData_sig1.dat"
TRAIN16_SIG2 = DATA_DIR / "upd2_n4096l_fw16_PartialTrainData_sig2.dat"

# Filter-width 32, signature 1 & 2
TRAIN32_SIG1 = DATA_DIR / "upd2_n4096l_fw32_PartialTrainData_sig1.dat"
TRAIN32_SIG2 = DATA_DIR / "upd2_n4096l_fw32_PartialTrainData_sig2.dat"

# And your source-term data files
TRAIN16_SOURCE = DATA_DIR / "upd_n4096l_fw16_PartialTrainData_sourceterm.dat"
TRAIN32_SOURCE = DATA_DIR / "upd_n4096l_fw32_PartialTrainData_sourceterm.dat"


# Model hyperparameters
LEARNING_RATE  = 1e-3
BATCH_SIZE     = 256
EPOCHS         = 100
NEURONS        = 50

# TensorBoard logs
LOG_DIR        = BASE / "logs"
