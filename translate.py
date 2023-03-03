import torch
from transformers import BertTokenizer
from utils import Seq2SeqTransformer
from config import EMB_SIZE, FFN_HID_DIM, NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS
from paths import *


class Translator:
    def __init__(self, src_lang, tgt_lang):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        path2ru_vocab = tokenizers_dict[src_lang]
        path2tt_vocab = tokenizers_dict[tgt_lang]
        self.tt_tokenizer = BertTokenizer(path2tt_vocab, sep_token='[EOS]', cls_token='[BOS]', strip_accents=False)
        self.ru_tokenizer = BertTokenizer(path2ru_vocab, sep_token='[EOS]', cls_token='[BOS]', strip_accents=False)

        path2model = models_dict[(src_lang, tgt_lang)]
        SRC_VOCAB_SIZE = len(self.ru_tokenizer)
        TGT_VOCAB_SIZE = len(self.tt_tokenizer)

        self.model = Seq2SeqTransformer(NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, EMB_SIZE, SRC_VOCAB_SIZE,
                                        TGT_VOCAB_SIZE, FFN_HID_DIM)

        self.model.load_state_dict(torch.load(path2model, map_location=self.device))

        self.model.eval()

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones((sz, sz), device=self.device)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def encode(self, model, src, src_mask):
        return model.transformer_encoder(model.positional_encoding(
            model.src_tok_emb(src)), src_mask)

    def decode(self, model, tgt, memory, tgt_mask=None):
        return model.transformer_decoder(model.positional_encoding(model.tgt_tok_emb(tgt)), memory, tgt_mask)

    def greedy_decode(self, model, src, src_mask, max_len, start_symbol):
        src = src.to(self.device)
        src_mask = src_mask.to(self.device)

        memory = self.encode(model, src, src_mask)
        ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(
            self.device)
        for i in range(max_len - 1):
            memory = memory.to(self.device)
            tgt_mask = (self.generate_square_subsequent_mask(ys.size(0))
                        .type(torch.bool)).to(self.device)
            out = self.decode(model, ys, memory, tgt_mask)
            out = out.transpose(0, 1)
            prob = model.generator(out[:, -1])
            _, next_word = torch.max(prob, dim=1)
            next_word = next_word.item()

            ys = torch.cat([ys,
                            torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=0)
            if next_word == 3:
                break
        return ys

    def translate(self, src_text):
        tokens = self.ru_tokenizer(src_text).input_ids
        num_tokens = len(tokens)
        src = (torch.LongTensor(tokens).reshape(num_tokens, 1))
        src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool)
        tgt_tokens = self.greedy_decode(self.model, src, src_mask, max_len=num_tokens + 5, start_symbol=2).flatten()
        return self.tt_tokenizer.decode(tgt_tokens).replace('[BOS] ', '').replace(' [EOS]', '')
