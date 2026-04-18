import unittest

import torch
from transformers import BertConfig
from transformers.models.bert.modeling_bert import BertSelfAttention

from function_handler import BertSelfAttentionWithAproximation


class BertAttentionCompatTests(unittest.TestCase):
    def setUp(self):
        self.config = BertConfig()
        self.hidden = torch.randn(2, 3, self.config.hidden_size)
        self.cache = (
            torch.randn(
                2,
                self.config.num_attention_heads,
                2,
                self.config.hidden_size // self.config.num_attention_heads,
            ),
            torch.randn(
                2,
                self.config.num_attention_heads,
                2,
                self.config.hidden_size // self.config.num_attention_heads,
            ),
        )

    def _make_attn(self):
        attn = BertSelfAttentionWithAproximation(
            self.config,
            degree=2,
            lower_bound=-4,
        )
        attn.eval()
        return attn

    def test_supports_legacy_positional_call_with_encoder_mask_slot(self):
        attn = self._make_attn()
        outputs = attn(self.hidden, None, None, None, None, self.cache, True)
        self.assertEqual(outputs[0].shape, (2, 3, self.config.hidden_size))
        self.assertEqual(outputs[1].shape[0], 2)

    def test_supports_legacy_positional_call_without_encoder_mask_slot(self):
        attn = self._make_attn()
        outputs = attn(self.hidden, None, None, None, self.cache, True)
        self.assertEqual(outputs[0].shape, (2, 3, self.config.hidden_size))
        self.assertEqual(outputs[1].shape[0], 2)

    def test_supports_legacy_positional_call_without_cache(self):
        attn = self._make_attn()
        outputs = attn(self.hidden, None, None, None, None, False)
        self.assertEqual(len(outputs), 1)
        self.assertEqual(outputs[0].shape, (2, 3, self.config.hidden_size))

    def test_falls_back_when_base_init_does_not_accept_layer_idx(self):
        original_init = BertSelfAttention.__init__

        def old_style_init(module_self, config, position_embedding_type=None):
            return original_init(
                module_self,
                config,
                position_embedding_type=position_embedding_type,
            )

        BertSelfAttention.__init__ = old_style_init
        try:
            attn = BertSelfAttentionWithAproximation(
                self.config,
                degree=2,
                lower_bound=-4,
                position_embedding_type="absolute",
                layer_idx=7,
            )
            outputs = attn(self.hidden, output_attentions=True)
            self.assertEqual(attn.layer_idx, 7)
            self.assertEqual(attn.position_embedding_type, "absolute")
            self.assertEqual(outputs[0].shape, (2, 3, self.config.hidden_size))
            self.assertEqual(outputs[1].shape[0], 2)
        finally:
            BertSelfAttention.__init__ = original_init


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
