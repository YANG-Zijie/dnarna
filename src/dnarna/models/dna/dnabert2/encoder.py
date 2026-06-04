import torch
from transformers import AutoModel, AutoTokenizer

from dnarna.models.shared.encoders.base_seq_encoder import BaseSeqEncoder, mean_pool


class DNABERT2Encoder(BaseSeqEncoder):
    def __init__(self, backbone: str = "zhihan1996/DNABERT-2-117M", **kwargs) -> None:
        super().__init__(**kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(backbone, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(backbone, trust_remote_code=True).to(
            self.device
        )
        self.model.eval()

    def encode_seq_batch(self, seqs: list[str]) -> torch.Tensor:
        enc = self.tokenizer(
            seqs,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        for k in enc:
            enc[k] = enc[k].to(self.device)
        with torch.no_grad():
            outputs = self.model(**enc)
        if hasattr(outputs, "last_hidden_state"):
            hidden = outputs.last_hidden_state  # standard HF output
        elif isinstance(outputs, tuple) and len(outputs) > 0:
            hidden = outputs[0]  # some DNABERT-2 variants return tuple
        else:
            raise TypeError(
                "Unexpected DNABERT-2 output type; expected BaseModelOutput or tuple"
            )
        pooled = mean_pool(hidden, enc["attention_mask"]).to(self.dtype).to("cpu")
        return pooled
