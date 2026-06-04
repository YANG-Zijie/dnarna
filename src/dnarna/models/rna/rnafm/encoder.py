import fm
import torch

from dnarna.models.shared.encoders.base_seq_encoder import BaseSeqEncoder, mean_pool

_RNAFM_MAX_TOKENS = 1024  # model adds BOS/EOS, so usable sequence length is <= 1022


class RNAFMEncoder(BaseSeqEncoder):
    """RNA-FM official encoder (pip install rna-fm).

    Parameters
    - variant: "rna" (RNA-FM, 640-d) or "mrna" (mRNA-FM, 1280-d; requires codon-aligned sequences)
    - replace_T_with_U: map DNA-style 'T' to RNA 'U' before tokenization
    """

    def __init__(
        self,
        variant: str = "rna",
        replace_T_with_U: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        if variant == "rna":
            self.model, self.alphabet = fm.pretrained.rna_fm_t12()
        elif variant == "mrna":
            self.model, self.alphabet = fm.pretrained.mrna_fm_t12()
        else:
            raise ValueError("variant must be 'rna' or 'mrna'")
        self.batch_converter = self.alphabet.get_batch_converter()
        self.model = self.model.to(self.device)
        self.model.eval()
        self.variant = variant
        self.replace_T_with_U = replace_T_with_U

        max_tokens = getattr(self.model, "max_positions", None)
        if max_tokens is None:
            max_tokens = getattr(
                getattr(self.model, "args", None), "max_positions", None
            )
        self._model_max_tokens = int(max_tokens) if max_tokens else _RNAFM_MAX_TOKENS
        # account for special tokens (CLS/EOS) added by the batch converter
        self._model_max_seq_len = max(1, self._model_max_tokens - 2)
        if self.max_length and self.max_length > 0:
            self._effective_max_len = min(self.max_length, self._model_max_seq_len)
        else:
            self._effective_max_len = self._model_max_seq_len

    def _prep(self, seqs: list[str]) -> list[tuple[str, str]]:
        xs: list[tuple[str, str]] = []
        for i, s in enumerate(seqs):
            s = str(s).upper()
            if self.replace_T_with_U:
                s = s.replace("T", "U")
            # keep common tokens: A,U,C,G,N and '-' (as in official examples)
            s = "".join([c for c in s if c in "AUCGN-"])
            # truncate by max_length; for mRNA ensure codon alignment
            max_len = self._effective_max_len
            if self.variant == "mrna":
                # keep length as multiple of 3 first, then apply max_length (also multiple of 3)
                if len(s) % 3 != 0:
                    s = s[: len(s) - (len(s) % 3)]
                if max_len:
                    L = max_len - (max_len % 3)
                    if L <= 0:
                        L = 3
                    s = s[:L]
            else:
                if max_len:
                    s = s[:max_len]
            if not s:
                s = "A"
            xs.append((f"RNA{i}", s))
        return xs

    def encode_seq_batch(self, seqs: list[str]) -> torch.Tensor:
        data = self._prep(seqs)
        _, _, tokens = self.batch_converter(data)
        tokens = tokens.to(self.device)
        with torch.no_grad():
            out = self.model(tokens, repr_layers=[12])
        # token representations at layer 12: [B, L, H]
        reps = out["representations"][12]
        # build attention mask: exclude padding/CLS/EOS if defined
        mask = tokens != getattr(self.alphabet, "padding_idx", 0)
        if hasattr(self.alphabet, "cls_idx"):
            mask = mask & (tokens != self.alphabet.cls_idx)
        if hasattr(self.alphabet, "eos_idx"):
            mask = mask & (tokens != self.alphabet.eos_idx)
        pooled = mean_pool(reps, mask).to(self.dtype).to("cpu")
        return pooled
