# VeriRel
+VeriRel: Verification Feedback to Enhance Document Retrieval for Scientific Fact Checking


## MultiVerS settings
Please use the provided code from [MultiVerS](https://github.com/dwadden/multivers) to train V_MultiVerS. 

For different negative sampling strategies, we have only made minor changes in data_train.py for adding new trainset frameworks. 

To force the model output probabilities, we have made minor changes to predict.py. We add an argument 'format', where the model predicts the result as set to 'result' in inference, and produces probabilities for subsequent calculating ComboScorer as set to 'prob'. The changed versions are presented in the folder of V_MultiVerS.

## T5-3B settings
For reproducibility of T5-3B reranker, please use [PyGaggle](https://github.com/castorini/pygaggle) with the provided code in the folder.

## Verification Evaluation
See  [SCIVER](https://github.com/allenai/scifact) shared task.


## Experiment settings
All training experiments set the random seed to 27 in our study, using a single NVIDIA A100 GPU.
