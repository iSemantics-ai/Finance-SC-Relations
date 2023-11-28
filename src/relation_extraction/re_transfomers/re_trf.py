from transformers import PreTrainedModel, AutoModel
from torch import nn
import torch

class RE_Transformers(PreTrainedModel):
    def __init__(self, config, load_base=False):
        """
        Initializes a custom transformer-based model for relation extraction.

        Args:
            config (transformers.PretrainedConfig): The configuration for the pre-trained transformer model.
        """
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        self.model = AutoModel.from_config(config) if not load_base\
                    else AutoModel.from_pretrained(self.config.base_model, config=config)
        self.classification_layer = nn.Linear(config.hidden_size * 2, config.num_labels)
        # Initialize weights and apply final processing
        self.post_init()
    
    def forward(self,
        input_ids=None,
        attention_mask=None, 
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_state= None,
        encoder_attention = None,
        Q=None,
        e1_e2_start=None):
        """
        Runs forward pass of the custom transformer-based model for relation extraction.

        Args:
            input_ids (torch.Tensor): The input token IDs.
            attention_mask (torch.Tensor): The attention mask for the input.
            token_type_ids (torch.Tensor): The token type IDs for the input.
            position_ids (torch.Tensor): The position IDs for the input.
            head_mask (torch.Tensor): The head mask for the input.
            inputs_embeds (torch.Tensor): The input embeddings.
            encoder_hidden_state (torch.Tensor): The encoder hidden state.
            encoder_attention (torch.Tensor): The encoder attention.
            Q (torch.Tensor): The query tensor.
            e1_e2_start (torch.Tensor): The start positions of the two entity spans.
        
        Returns:
            classification_logits (torch.Tensor): The logits for classification.
        """
        sequence_output = self.model(
            input_ids, attention_mask=attention_mask
        ).last_hidden_state
         ### two heads: LM and blanks ###
        blankv1v2 = sequence_output[:, e1_e2_start, :]
        buffer = []
        for i in range(blankv1v2.shape[0]):  # iterate batch & collect
            v1v2 = blankv1v2[i, i, :, :]
            v1v2 = torch.cat((v1v2[0], v1v2[1]))
            buffer.append(v1v2)
        del blankv1v2
        v1v2 = torch.stack([a for a in buffer], dim=0)
        del buffer
        classification_logits = self.classification_layer(v1v2)
        return classification_logits