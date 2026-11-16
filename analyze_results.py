"""Create training plots and validation-only operating curves for synthetic TTS demo."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from config import CLASSES
from metrics import score_probabilities
from model import load_checkpoint
from train import dataset_for
from tune import probabilities_for_dataset
from types import SimpleNamespace


def run():
    Path('results').mkdir(exist_ok=True)
    training = {}
    for mode in ('baseline','prefix'):
        training[mode] = json.loads(Path(f'results/training_synthetic_{mode}.json').read_text())
    plt.figure(figsize=(7.3, 4.2))
    for mode in ('baseline', 'prefix'):
        history = training[mode]['history']
        plt.plot([h['epoch'] for h in history], [100*h['validation_accuracy'] for h in history], label=mode)
    plt.xlabel('Epoch')
    plt.ylabel('Validation accuracy (%)')
    plt.title('Synthetic speech: LSTM validation accuracy')
    plt.legend()
    plt.grid(alpha=.25)
    plt.tight_layout()
    plt.savefig('results/synthetic_training_accuracy.png', dpi=165)
    plt.close()

    args = SimpleNamespace(dataset='synthetic', synthetic_path='data/synthetic_tts.pt')
    valid_set, test_set = dataset_for(args, 'validation'), dataset_for(args, 'test')
    grid = np.arange(.20, 1, .05).tolist() + [.97, .98, .99, .995]
    all_curves = {}
    trade_fig,trade_ax = plt.subplots(figsize=(7.3, 4.6))
    for mode in ('baseline','prefix'):
        model,_ = load_checkpoint(f'models/synthetic_{mode}.pt',torch.device('cpu'))
        probs,labels = probabilities_for_dataset(model,torch.device('cpu'),valid_set,64)
        curve = []
        for threshold in grid:
            metrics = score_probabilities(probs,labels,threshold,3,300,2)
            curve.append({'threshold': float(threshold),
                          'early_keyword_recall': metrics['early_keyword_recall'],
                          'negative_clip_trigger_rate': metrics['early_negative_trigger_rate'],
                          'p50_emission_ms': metrics['p50_emission_ms']})
        all_curves[mode] = curve
        trade_ax.plot([100*p['negative_clip_trigger_rate'] for p in curve],
                 [100*p['early_keyword_recall'] for p in curve], 'o-', markersize=3, label=mode)
        if mode == 'prefix':
            test_probs,test_labels = probabilities_for_dataset(model,torch.device('cpu'),test_set,64)
            fixed = np.asarray(test_probs)[:,-1,:].argmax(-1)
            conf = confusion_matrix(test_labels,fixed,labels=list(range(len(CLASSES))))
            fig,ax = plt.subplots(figsize=(8.5,7.2))
            ConfusionMatrixDisplay(confusion_matrix=conf,display_labels=CLASSES).plot(ax=ax,xticks_rotation=45,values_format='d',colorbar=False)
            ax.set_title('Synthetic held-out test: prefix model (full-window decisions)')
            fig.tight_layout()
            fig.savefig('results/synthetic_test_confusion.png',dpi=140)
            plt.close(fig)
    trade_ax.set_xlabel('Negative clip trigger rate (%)')
    trade_ax.set_ylabel('Early keyword recall (%)')
    trade_ax.set_title('Validation-only early-exit trade-off (synthetic TTS)')
    trade_ax.grid(alpha=.25)
    trade_ax.legend()
    trade_fig.tight_layout()
    trade_fig.savefig('results/synthetic_validation_tradeoff.png',dpi=165)
    plt.close(trade_fig)
    Path('results/synthetic_validation_curve.json').write_text(json.dumps(all_curves,indent=2))
    print('Created training, trade-off, and confusion plots.')


if __name__ == '__main__':
    run()
