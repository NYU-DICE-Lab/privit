import jax
import jax.numpy as jnp
import flax
from flax import traverse_util
from typing import Tuple
from transformers.models.vit.configuration_vit import ViTConfig

def accuracy(predictions, labels):
    if predictions.shape != labels.shape:
        raise ValueError("Predictions and labels must have the same shape")

    correct_predictions = jnp.sum(predictions == labels)
    total_predictions = predictions.size

    return correct_predictions / total_predictions


def print_named_params(model):
    params = model.params
    for path, value in traverse_util.flatten_dict(params).items():
        print(path, value.shape)


def get_config(dataset):
    if dataset == 'tiny_imagenet':
        config = ViTConfig(
            image_size=64,  # The size of input images
            patch_size=4,  # Size of patches to be extracted from the images
            num_channels=3,  # Number of channels of the input images
            num_labels=200,  # Number of labels for classification task
            hidden_size=192,  # Dimensionality of the encoder layers and the pooler layer
            num_hidden_layers=9,  # Number of hidden layers in the Transformer encoder
            num_attention_heads=12,  # Number of attention heads for each attention layer in the Transformer encoder
            intermediate_size=384,  # Dimensionality of the "intermediate" layer in the Transformer encoder
            hidden_act="gelu",  # The non-linear activation function in the encoder and pooler
            hidden_dropout_prob=0.,  # The dropout probabilitiy for all fully connected layers in the embeddings, encoder, and pooler
            attention_probs_dropout_prob=0.1,  # The dropout ratio for the attention probabilities
            qkv_bias=False,
            layer_norm_eps = 1e-05
        )
    elif dataset == 'cifar100':
        config = ViTConfig(
            image_size=32,  # The size of input images
            patch_size=4,  # Size of patches to be extracted from the images
            num_channels=3,  # Number of channels of the input images
            num_labels=100,  # Number of labels for classification task
            hidden_size=256,  # Dimensionality of the encoder layers and the pooler layer
            num_hidden_layers=7,  # Number of hidden layers in the Transformer encoder
            num_attention_heads=4,  # Number of attention heads for each attention layer in the Transformer encoder
            intermediate_size=512,  # Dimensionality of the "intermediate" layer in the Transformer encoder
            hidden_act="gelu",  # The non-linear activation function in the encoder and pooler
            hidden_dropout_prob=0.,  # The dropout probabilitiy for all fully connected layers in the embeddings, encoder, and pooler
            attention_probs_dropout_prob=0.1,  # The dropout ratio for the attention probabilities
            qkv_bias=False,
            layer_norm_eps = 1e-05
        )
    else:
        config = ViTConfig(
        image_size=32,  # The size of input images
        patch_size=4,  # Size of patches to be extracted from the images
        num_channels=3,  # Number of channels of the input images
        num_labels=10,  # Number of labels for classification task
        hidden_size=256,  # Dimensionality of the encoder layers and the pooler layer
        num_hidden_layers=7,  # Number of hidden layers in the Transformer encoder
        num_attention_heads=4,  # Number of attention heads for each attention layer in the Transformer encoder
        intermediate_size=512,  # Dimensionality of the "intermediate" layer in the Transformer encoder
        hidden_act="gelu",  # The non-linear activation function in the encoder and pooler
        hidden_dropout_prob=0.,  # The dropout probabilitiy for all fully connected layers in the embeddings, encoder, and pooler
        attention_probs_dropout_prob=0.1,  # The dropout ratio for the attention probabilities
        qkv_bias=False,
        layer_norm_eps = 1e-05
        )
    return config

def pytorch_to_flax(state_dict,flax_model):
    flax_params = {}
    for name, param in state_dict.items():
        if 'projection' in name:
            if param.dim() == 4:
                param_data = param.detach().cpu().numpy()
                param_data = jnp.transpose(param_data,axes=(2,3,1,0))
            elif param.dim() == 1:  # Typically bias
                param_data = param.detach().cpu().numpy()
            else:
                print(f"Unhandled shape {param.shape} for {name}")
                continue
        else:
            param_data = param.detach().cpu().numpy()
        
        name_parts = name.split('.')
        
        if len(name_parts) > 3:
            if name_parts[0] == "vit" and name_parts[1] == "encoder" and name_parts[4] == "layernorm_before":
                if name_parts[5] == "weight":
                    flax_name = tuple(("vit", "encoder", "layer", name_parts[3], 'layernorm_before',"scale"))
                    flax_params[flax_name] = param_data
                elif name_parts[5] == "bias":
                    flax_name = tuple(("vit", "encoder", "layer", name_parts[3], 'layernorm_before',"bias")) # flax 
                    flax_params[flax_name] = param_data
            
            if name_parts[0] == "vit" and name_parts[1] == "encoder" and name_parts[4] == "layernorm_after":
                if name_parts[5] == "weight":
                    flax_name = tuple(("vit", "encoder", "layer", name_parts[3], 'layernorm_after',"scale"))

                    flax_params[flax_name] = param_data
                else:
                    flax_name = tuple(("vit", "encoder", "layer", name_parts[3], 'layernorm_after',"bias"))

                    flax_params[flax_name] = param_data
            
            if name_parts[0] == "vit" and name_parts[1] == "encoder" and name_parts[4] == "attention" and name_parts[5] == "attention":
                if name_parts[6] == 'query':
                    if name_parts[7] == 'weight':
                        flax_name = tuple(("vit", "encoder", "layer", name_parts[3], 'attention','attention','query',"kernel"))
                        flax_params[flax_name] = jnp.transpose(param_data)
                    else:
                        flax_name = tuple(("vit", "encoder", "layer", name_parts[3], 'attention','attention','query',"bias"))
                        flax_params[flax_name] = param_data

                if name_parts[6] == 'value':
                    if name_parts[7] == 'weight':
                        flax_name = tuple(("vit", "encoder", "layer", name_parts[3], 'attention','attention','value',"kernel"))
                        flax_params[flax_name] = jnp.transpose(param_data)
                    else:
                        flax_name = tuple(("vit", "encoder", "layer", name_parts[3], 'attention','attention','value',"bias"))
                        flax_params[flax_name] = param_data
                if name_parts[6] == 'key':
                    if name_parts[7] == 'weight':
                        flax_name = tuple(("vit", "encoder", "layer", name_parts[3], 'attention','attention','key',"kernel"))
                        flax_params[flax_name] = jnp.transpose(param_data)
                    else:
                        flax_name = tuple(("vit", "encoder", "layer", name_parts[3], 'attention','attention','key',"bias"))
                        flax_params[flax_name] = param_data
            if name_parts[0] == "vit" and name_parts[1] == "encoder" and name_parts[4] == "attention" and name_parts[5] == "output" and name_parts[6] == "dense":
                if name_parts[7] == "weight":
                    flax_name = tuple(('vit', 'encoder', 'layer',name_parts[3],'attention', 'output', 'dense', 'kernel'))
                    flax_params[flax_name] = jnp.transpose(param_data,(1,0))
                else:
                    flax_name = tuple(('vit', 'encoder', 'layer',name_parts[3],'attention', 'output', 'dense', 'bias'))
                    flax_params[flax_name] = param_data
                

            if name_parts[0] == "vit" and name_parts[1] == "encoder" and name_parts[4] == "intermediate" and name_parts[5] == "dense"and name_parts[6] == "weight":
                flax_name = tuple(("vit", "encoder", "layer", name_parts[3], 'intermediate','dense','kernel'))
                flax_params[flax_name] = jnp.transpose(param_data,(1,0))
            
            if name_parts[0] == "vit" and name_parts[1] == "encoder" and name_parts[4] == "intermediate" and name_parts[5] == "dense"and name_parts[6] == "bias":
                flax_name = tuple(("vit", "encoder", "layer", name_parts[3], 'intermediate','dense','bias'))
                flax_params[flax_name] = param_data

            if name_parts[0] == "vit" and name_parts[1] == "encoder" and name_parts[4] == "output" and name_parts[5] == "dense"and name_parts[6] == "weight":
                flax_name = tuple(("vit", "encoder", "layer", name_parts[3], 'output','dense','kernel'))
                flax_params[flax_name] = jnp.transpose(param_data)
            
            if name_parts[0] == "vit" and name_parts[1] == "encoder" and name_parts[4] == "output" and name_parts[5] == "dense"and name_parts[6] == "bias":
                flax_name = tuple(("vit", "encoder", "layer", name_parts[3], 'output','dense','bias'))
                flax_params[flax_name] = param_data
        
        if name_parts[0] == "vit" and name_parts[1] == "embeddings" and name_parts[2] == "cls_token":
            flax_name = tuple(('vit', 'embeddings', 'cls_token'))
            flax_params[flax_name] = param_data
        
        if name_parts[0] == "vit" and name_parts[1] == 'embeddings' and name_parts[2] == "position_embeddings":
            flax_name = tuple(('vit', 'embeddings', 'position_embeddings'))
            flax_params[flax_name] = param_data
        
        if name_parts[0] == "vit" and name_parts[1] =='embeddings' and name_parts[2] =="patch_embeddings" and name_parts[4] =="weight":
            flax_name = tuple(('vit', 'embeddings', 'patch_embeddings','projection','kernel'))
            flax_params[flax_name] = param_data
        
        if name_parts[0] == "vit" and name_parts[1] =='embeddings' and name_parts[2] =="patch_embeddings" and name_parts[4] =="bias":
            flax_name = tuple(('vit', 'embeddings', 'patch_embeddings','projection','bias'))
            flax_params[flax_name] = param_data
        
        if name_parts[0] == "classifier" and name_parts[1] =="weight":
            flax_name = tuple(('classifier', 'kernel'))
            flax_params[flax_name] = jnp.transpose(param_data,(1,0))
        if name_parts[0] == "classifier" and name_parts[1] =="bias":
            flax_name = tuple(('classifier', 'bias'))
            flax_params[flax_name] = param_data
        
        if name_parts[0] == "vit" and name_parts[1] =="layernorm" and name_parts[2] =="weight":
            flax_name = tuple(('vit','layernorm' ,'scale'))
            flax_params[flax_name] = param_data
        if name_parts[0] == "vit" and name_parts[1] =="layernorm" and name_parts[2] =="bias":
            flax_name = tuple(('vit','layernorm' ,'bias'))
            flax_params[flax_name] = param_data
    flax_params = flax.traverse_util.unflatten_dict(flax_params)
    flax_model.params = flax.core.unfreeze(flax.core.FrozenDict(flax_params))
    return flax_model


def get_infer_cipher(pytorch_model_state_dict):
    alpha_infer =[]
    beta_infer = []
    for name, param in pytorch_model_state_dict.items():
        if 'projection' not in name:
            param_data = param.detach().cpu().numpy()
        
        name_parts = name.split('.')
        if len(name_parts) > 3:

            if name_parts[0] == "vit" and name_parts[1] == "encoder" and name_parts[4] == "intermediate" and name_parts[5] == "intermediate_act_fn" and name_parts[6] == "alphas":
                alpha_infer.append(param_data)
            if name_parts[0] == "vit" and name_parts[1] == "encoder" and name_parts[4] == "attention" and name_parts[5] == "attention" and name_parts[6] == "betas" and name_parts[7] == "betas":
                beta_infer.append(param_data)
    return alpha_infer, beta_infer


def get_infer_cipher_mpc_vit(pytorch_model_state_dict):
    alpha_infer = []
    beta_infer = []
    for name, param in pytorch_model_state_dict.items():
        if 'conv' not in name:
            param_data = param.detach().cpu().numpy()
        name_parts = name.split('.')
        if len(name_parts) > 3:
            if name_parts[0] == "classifier" and name_parts[1] == "blocks" and name_parts[3] == "self_attn" and name_parts[4] == "alpha":
                alpha_infer.append(param_data)
            if name_parts[0] == "classifier" and name_parts[1] == "blocks" and name_parts[3] == "beta":
                beta_infer.append(param_data)
    return alpha_infer, beta_infer


def map_qkv_weights_to_flax(pytorch_qkv_weight):
    """
    This function assumes pytorch_qkv_weight has shape (C, 3 * C).
    It splits this into three (C, C) matrices for query, key, and value.
    """
    C = pytorch_qkv_weight.shape[0]
    third_dim = C // 3  # Assuming 3 for q, k, v (576,192)

    query_weight = pytorch_qkv_weight[:third_dim,:]
    key_weight = pytorch_qkv_weight[third_dim : 2 * third_dim,:]
    value_weight = pytorch_qkv_weight[2 * third_dim:,:]
    
    return query_weight, key_weight, value_weight

def pytorch_to_flax_mpcvit(pytorch_model_state_dict, flax_model):
    flax_params = {}
    for name, param in pytorch_model_state_dict.items():
        if 'conv' in name:
            if param.dim() == 4:
                param_data = param.detach().cpu().numpy()
                param_data = jnp.transpose(param_data,axes=(2,3,1,0))
            elif param.dim() == 1:  # Typically bias
                param_data = param.detach().cpu().numpy()
            else:
                print(f"Unhandled shape {param.shape} for {name}")
                continue
        else:
            param_data = param.detach().cpu().numpy()
        
        name_parts = name.split('.')
        
        if len(name_parts) > 3:
            if name_parts[0] == "classifier" and name_parts[1] == "blocks" and name_parts[3] == "beta":
                flax_name = tuple(("vit", "encoder", "layer", name_parts[2], 'intermediate','chooseActivation',"beta"))
                flax_params[flax_name] = param_data
            if name_parts[0] == "classifier" and name_parts[1] == "blocks" and name_parts[3] == "pre_norm":
                if name_parts[4] == "weight":
                    flax_name = tuple(("vit", "encoder", "layer", name_parts[2], 'layernorm_before',"scale"))
                    flax_params[flax_name] = param_data
                elif name_parts[4] == "bias":
                    flax_name = tuple(("vit", "encoder", "layer", name_parts[2], 'layernorm_before',"bias")) # flax 
                    flax_params[flax_name] = param_data
            
            if name_parts[0] == "classifier" and name_parts[1] == "blocks" and name_parts[3] == "norm1":
                if name_parts[4] == "weight":
                    flax_name = tuple(("vit", "encoder", "layer", name_parts[2], 'layernorm_after',"scale"))

                    flax_params[flax_name] = param_data
                else:
                    flax_name = tuple(("vit", "encoder", "layer", name_parts[2], 'layernorm_after',"bias"))

                    flax_params[flax_name] = param_data
            
            if name_parts[0] == "classifier" and name_parts[1] == "blocks" and name_parts[3] == "self_attn" and name_parts[4] == "qkv":
                weight = param.data.cpu().numpy()
                query_weight, key_weight, value_weight = map_qkv_weights_to_flax(weight)
                flax_name_key = tuple(("vit", "encoder", "layer", name_parts[2], 'attention','attention','key',"kernel"))
                flax_name_query = tuple(("vit", "encoder", "layer", name_parts[2], 'attention','attention','query',"kernel"))
                flax_name_value = tuple(("vit", "encoder", "layer", name_parts[2], 'attention','attention','value',"kernel"))
                flax_params[flax_name_key] = jnp.transpose(key_weight)
                flax_params[flax_name_query] = jnp.transpose(query_weight)
                flax_params[flax_name_value] = jnp.transpose(value_weight)
            
            if name_parts[0] == "classifier" and name_parts[1] == "blocks" and name_parts[3] == "self_attn" and name_parts[4] == "alpha":
                flax_name = tuple(('vit', 'encoder', 'layer', name_parts[2], 'attention', 'attention','chooseAttention', 'alpha'))
                flax_params[flax_name] = param_data
                # alpha_infer.append(param_data)
                # flax_params[flax_name] = jnp.ones((1,12,1,1))
            if name_parts[0] == "classifier" and name_parts[1] == "blocks" and name_parts[3] == "self_attn" and name_parts[4] == "proj":
                if name_parts[5] == "weight":
                    flax_name = tuple(('vit', 'encoder', 'layer',name_parts[2],'attention', 'output', 'dense', 'kernel'))
                    flax_params[flax_name] = jnp.transpose(param_data,(1,0))
                else:
                    flax_name = tuple(('vit', 'encoder', 'layer',name_parts[2],'attention', 'output', 'dense', 'bias'))
                    flax_params[flax_name] = param_data
                

            if name_parts[0] == "classifier" and name_parts[1] == "blocks" and name_parts[3] == "linear1" and name_parts[4] == "weight":
                flax_name = tuple(("vit", "encoder", "layer", name_parts[2], 'intermediate','dense','kernel'))
                flax_params[flax_name] = jnp.transpose(param_data,(1,0))
            
            if name_parts[0] == "classifier" and name_parts[1] == "blocks" and name_parts[3] == "linear1" and name_parts[4] == "bias":
                flax_name = tuple(("vit", "encoder", "layer", name_parts[2], 'intermediate','dense','bias'))
                flax_params[flax_name] = param_data

            if name_parts[0] == "classifier" and name_parts[1] == "blocks" and name_parts[3] == "linear2" and name_parts[4] == "weight":
                flax_name = tuple(("vit", "encoder", "layer", name_parts[2], 'output','dense','kernel'))
                flax_params[flax_name] = jnp.transpose(param_data)
            
            if name_parts[0] == "classifier" and name_parts[1] == "blocks" and name_parts[3] == "linear2" and name_parts[4] == "bias":
                flax_name = tuple(("vit", "encoder", "layer", name_parts[2], 'output','dense','bias'))
                flax_params[flax_name] = param_data
        
        if name_parts[0] == "classifier" and name_parts[1] == "class_emb":
            flax_name = tuple(('vit', 'embeddings', 'cls_token'))
            flax_params[flax_name] = param_data
        
        if name_parts[0] == "classifier" and name_parts[1] == "positional_emb":
            flax_name = tuple(('vit', 'embeddings', 'position_embeddings'))
            flax_params[flax_name] = param_data
        
        if name_parts[0] == "tokenizer" and name_parts[4] =="weight":
            flax_name = tuple(('vit', 'embeddings', 'patch_embeddings','projection','kernel'))
            flax_params[flax_name] = param_data
        
        if name_parts[0] == "tokenizer" and name_parts[4] =="bias":
            flax_name = tuple(('vit', 'embeddings', 'patch_embeddings','projection','bias'))
            flax_params[flax_name] = param_data
        
        if name_parts[0] == "classifier" and name_parts[1] =="fc" and name_parts[2] =="weight":
            flax_name = tuple(('classifier', 'kernel'))
            flax_params[flax_name] = jnp.transpose(param_data,(1,0))
        if name_parts[0] == "classifier" and name_parts[1] =="fc" and name_parts[2] =="bias":
            flax_name = tuple(('classifier', 'bias'))
            flax_params[flax_name] = param_data
        
        if name_parts[0] == "classifier" and name_parts[1] =="norm" and name_parts[2] =="weight":
            flax_name = tuple(('vit','layernorm' ,'scale'))
            flax_params[flax_name] = param_data
        if name_parts[0] == "classifier" and name_parts[1] =="norm" and name_parts[2] =="bias":
            flax_name = tuple(('vit','layernorm' ,'bias'))
            flax_params[flax_name] = param_data
    flax_params = flax.traverse_util.unflatten_dict(flax_params)
    flax_model.params = flax.core.unfreeze(flax.core.FrozenDict(flax_params))
    return flax_model

