| model | max_length | threshold | threshold_source | n_processed | n_skipped | coverage | positive_prevalence | all_negative_accuracy_baseline | accuracy | precision | recall | f1 | mcc | roc_auc_mean | pr_auc_mean | top1_acc | topk_recall |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ModernBERT (len=2048) | 2048 | 0.5000 | result.json | 50357 | 5 | 0.9999 | 0.2086 | 0.7914 | 0.883 | 0.667 | 0.875 | 0.757 | 0.693 | 0.9275 | 0.8128 | 0.897 | 0.880 |
| BAAI-bge-m3 (len=1024) | 1024 | 0.4000 | result.json | 50347 | 15 | 0.9997 | 0.2086 | 0.7914 | 0.815 | 0.534 | 0.892 | 0.668 | 0.585 | 0.9201 | 0.7910 | 0.891 | 0.874 |
| DeBERTa-v3-large (len=512) | 512 | 0.2000 | result.json | 45081 | 5281 | 0.8951 | 0.2233 | 0.7767 | 0.694 | 0.417 | 0.936 | 0.577 | 0.466 | 0.9126 | 0.7749 | 0.839 | 0.825 |
