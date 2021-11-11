# Oracle-Baseline

Implent of Oracle Base line and Lea-3 Baseline

## Oracle

**Oracle :** This model is used to obtain an oracle with a greedy algorithm similar to Nallapati et al. (2017)
, which treats the utterances that maximize the ROUGE-2 as a summary.

```shell
python train_oracle.py
```

Please remember to change file path in this code , qwq. You can change the number of the oracle you want by changing `summary_size` in `greedy_selection`.

## Lead-3

**Lead-3**: This model simply selects the first three sentences in a document as summary.

```
python train_lead.py
```

Please remember to change file path in this code , qwq. You can change the number of the oracle you want by changing `lead_size` in `lead_selection`.

## Calculate Rouge

You can use `cal_rouge.py` to calculate rouge scores here.

## Requirements

```
rouge==1.0.1
nltk==3.6.5
regex == 2021.8.3

```











