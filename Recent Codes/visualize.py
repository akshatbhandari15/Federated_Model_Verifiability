import glob
import matplotlib.pyplot as plt
import pandas as pd

from pathlib import Path

def visualize_phase0(niids):
    iid_files = glob.glob('outputs/phase0_iid_*.xlsx')
    for f in iid_files:
        df = pd.read_excel(f)
        plt.plot(df['test_accuracy'], label = Path(f).stem)
        plt.show()