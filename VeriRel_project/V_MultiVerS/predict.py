from tqdm import tqdm
import argparse
from pathlib import Path
import torch
from model import MultiVerSModel
from data import get_dataloader
import util
import argparse


parser = argparse.ArgumentParser(description="Process some arguments.")
parser.add_argument('--checkpoint_path', type=str, help='Path to the checkpoint file')
parser.add_argument('--input_file', type=str, help='Path to the input file')
parser.add_argument('--corpus_file', type=str, default="dataset/scifact/corpus.jsonl", help='Path to the corpus file')
parser.add_argument('--output_file', type=str, help='Path to the output file')
parser.add_argument('--format', type=str, default='result', help='Format of the output')
parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
parser.add_argument('--device', type=int, default=0, help='Device ID')
parser.add_argument('--num_workers', type=int, default=4, help='Number of workers')
parser.add_argument('--no_nei', action='store_true', help='Flag to disable NEI option')
parser.add_argument('--force_rationale', action='store_true', help='Flag to force rationale usage')

args = parser.parse_args()



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MultiVerSModel.load_from_checkpoint(args.checkpoint_path)
_ = model.to(device)
print(device)
model.eval()
model.freeze()

hparams = model.hparams["hparams"]
del hparams.precision
# Don' use 16-bit precision during evaluation.
for k, v in vars(args).items():
    if hasattr(hparams, k):
        setattr(hparams, k, v)



def format_predictions(args, predictions_all):
    # Need to get the claim ID's from the original file, since the data loader
    # won't have a record of claims for which no documents were retireved.
    claims = util.load_jsonl(args.input_file)
    claim_ids = [x["id"] for x in claims]
    assert len(claim_ids) == len(set(claim_ids))

    formatted = {claim: {} for claim in claim_ids}

    if args.format == 'prob':
        # Dict keyed by claim.
        for prediction in predictions_all:
            # If it's NEI, skip it.
            if prediction["predicted_label"] == "NEI":
                formatted_entry = {
                prediction["abstract_id"]: {
                    "label": prediction["predicted_label"],
                    "sentences": prediction["predicted_rationale"],
                    "probability": prediction['label_probs']
                }
            }
                # continue
            # else:
            # Add prediction.
            formatted_entry = {
                prediction["abstract_id"]: {
                    "label": prediction["predicted_label"],
                    "sentences": prediction["predicted_rationale"],
                    "probability": prediction['label_probs']
                }
            }
            formatted[prediction["claim_id"]].update(formatted_entry)
    elif args.format == 'result':
        # Dict keyed by claim.
        for prediction in predictions_all:
            # If it's NEI, skip it.
            if prediction["predicted_label"] == "NEI":
                continue
            # else:
            # Add prediction.
            formatted_entry = {
                prediction["abstract_id"]: {
                    "label": prediction["predicted_label"],
                    "sentences": prediction["predicted_rationale"]
                }
            }
            formatted[prediction["claim_id"]].update(formatted_entry) 

    # Convert to jsonl.
    res = []
    for k, v in formatted.items():
        to_append = {"id": k, "evidence": v}
        res.append(to_append)

    return res

# formatted = format_predictions(args, predictions_all)
# outname = Path(args.output_file)
# util.write_jsonl(formatted, outname)

if __name__ == '__main__':
    dataloader = get_dataloader(args)
    predictions_all = []

    for batch in tqdm(dataloader):
        preds_batch = model.predict(batch, args.force_rationale)
        predictions_all.extend(preds_batch)

    formatted = format_predictions(args, predictions_all)
    outname = Path(args.output_file)
    util.write_jsonl(formatted, outname)