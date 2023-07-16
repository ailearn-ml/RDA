# Code and Appendix for RDA.

Code and Appendix for "Imbalanced Label Distribution Learning" in AAAI 2023.

## Requirements

- Python >= 3.6
- PyTorch >= 1.10
- NumPy >= 1.13.3
- Scikit-learn >= 0.20

## Running the scripts

To train and test the RDA model in the terminal, use:

```bash
$ python run_RDA.py --dataset sample_data --hidden_dim 100 --lambda1 0.1 --lambda2 0.1 --lambda3 0.1 --max_epoch 300 --batch_size 50 --lr 0.001 --valid_size 20 --device cuda:0 --seed 0
```
