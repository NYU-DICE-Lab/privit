import jax
import jax.numpy as jnp
from jax import random
import flax
import flax.linen as nn
import jax.nn as jnn
from flax.linen.dtypes import promote_dtype
from transformers.models.vit.modeling_flax_vit import FlaxViTLayerCollection, FlaxViTEncoder, FlaxViTModule, FlaxViTForImageClassificationModule, FlaxViTPreTrainedModel, FlaxViTPooler, FlaxViTOutput, FlaxViTEmbeddings, FlaxViTSelfOutput
from transformers.models.vit.configuration_vit import ViTConfig
from transformers.modeling_flax_outputs import FlaxBaseModelOutput

class ChooseAttention(nn.Module):
    layer: int
    config: ViTConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        infer_beta = self.config.infer_beta[self.layer][0]
        self.true_indices = jnp.where(infer_beta, size=(self.config.beta_sizes[self.layer][0]))
        self.false_indices = jnp.where(infer_beta<0.5, size=(591-self.config.beta_sizes[self.layer][0]))

    def __call__(self, attn_weights):
        selected_attn_weights = attn_weights[:, self.true_indices[0], self.true_indices[1], :]
        selected_attn_weights_false = attn_weights[:,self.false_indices[0],self.false_indices[1],:]
        # Modify the selected attention weights based on your custom function
        modified_attn_weights = jax.nn.softmax(selected_attn_weights, axis=-1)
        modified_attn_weights_false = jnp.square(selected_attn_weights_false)/197

        # Update the original attention weights at the specified indices
        updated_attn_weights = attn_weights.at[:, self.true_indices[0], self.true_indices[1], :].set(modified_attn_weights)
        updated_attn_weights = updated_attn_weights.at[:, self.false_indices[0],self.false_indices[1],:].set(modified_attn_weights_false)

        return updated_attn_weights

class CustomFlaxViTSelfAttention(nn.Module):
    layer:int
    config: ViTConfig
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation
    
    def setup(self):
        if self.config.hidden_size % self.config.num_attention_heads != 0:
            raise ValueError(
                "`config.hidden_size`: {self.config.hidden_size} has to be a multiple of `config.num_attention_heads`:"
                " {self.config.num_attention_heads}"
            )

        self.query = nn.Dense(
            self.config.hidden_size,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.variance_scaling(
                self.config.initializer_range**2, mode="fan_in", distribution="truncated_normal"
            ),
            use_bias=self.config.qkv_bias,
        )
        self.key = nn.Dense(
            self.config.hidden_size,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.variance_scaling(
                self.config.initializer_range**2, mode="fan_in", distribution="truncated_normal"
            ),
            use_bias=self.config.qkv_bias,
        )
        self.value = nn.Dense(
            self.config.hidden_size,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.variance_scaling(
                self.config.initializer_range**2, mode="fan_in", distribution="truncated_normal"
            ),
            use_bias=self.config.qkv_bias,
        )
        self.chooseAttention = ChooseAttention(self.layer,self.config)
    
    def __call__(self, hidden_states, deterministic: bool = True, output_attentions: bool = False):
        
        head_dim = self.config.hidden_size // self.config.num_attention_heads

        query_states = self.query(hidden_states).reshape(
            hidden_states.shape[:2] + (self.config.num_attention_heads, head_dim)
        )
        value_states = self.value(hidden_states).reshape(
            hidden_states.shape[:2] + (self.config.num_attention_heads, head_dim)
        )
        key_states = self.key(hidden_states).reshape(
            hidden_states.shape[:2] + (self.config.num_attention_heads, head_dim)
        )

        dropout_rng = None
        if not deterministic and self.config.attention_probs_dropout_prob > 0.0:
            dropout_rng = self.make_rng("dropout")

        query, key = promote_dtype(query_states, key_states, dtype=self.dtype)
        dtype = query.dtype

        assert query.ndim == key.ndim, 'q, k must have same rank.'
        assert query.shape[:-3] == key.shape[:-3], 'q, k batch dims must match.'
        assert query.shape[-2] == key.shape[-2], 'q, k num_heads must match.'
        assert query.shape[-1] == key.shape[-1], 'q, k depths must match.'

        # calculate attention matrix
        depth = query.shape[-1]
        query = query / jnp.sqrt(depth).astype(dtype)
        attn_weights = jnp.einsum(
            '...qhd,...khd->...hqk', query, key, precision=None
        )
        attn_weights = self.chooseAttention(attn_weights)
        if not deterministic and self.config.attention_probs_dropout_prob > 0.0:
            keep_prob = 1.0 - self.config.attention_probs_dropout_prob
            if True:
                dropout_shape = tuple([1] * (key.ndim - 2)) + attn_weights.shape[-2:]
                keep = random.bernoulli(dropout_rng, keep_prob, dropout_shape)  
            else:
                keep = random.bernoulli(dropout_rng, keep_prob, attn_weights.shape) 
            multiplier = keep.astype(dtype) / jnp.asarray(keep_prob, dtype=dtype)
            attn_weights = attn_weights * multiplier

        attn_output = jnp.einsum("...hqk,...khd->...qhd", attn_weights, value_states)
        attn_output = attn_output.reshape(attn_output.shape[:2] + (-1,))

        outputs = (attn_output, attn_weights) if output_attentions else (attn_output,)
        return outputs



class CustomFlaxViTAttention(nn.Module):
    layer: int
    config: ViTConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.attention = CustomFlaxViTSelfAttention(self.layer, self.config, dtype=self.dtype)
        self.output = FlaxViTSelfOutput(self.config, dtype=self.dtype)

    def __call__(self, hidden_states, deterministic=True, output_attentions: bool = False):
        attn_outputs = self.attention(hidden_states, deterministic=deterministic, output_attentions=output_attentions)
        attn_output = attn_outputs[0]
        hidden_states = self.output(attn_output, hidden_states, deterministic=deterministic)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_outputs[1],)

        return outputs


class ChooseActivation(nn.Module):
    config: ViTConfig
    layer: int
    dtype: jnp.dtype = jnp.float32
    
    def setup(self):
        infer_alpha = self.config.infer_alpha[self.layer][0]
        self.true_indices = jnp.where(infer_alpha, size = self.config.alpha_sizes[self.layer][0])
        
    def __call__(self, hidden_states):
        selected_hidden_states = hidden_states[:, self.true_indices[0], :] # tokenwise
        
        activated_hidden_states = nn.gelu(selected_hidden_states)
        # activated_hidden_states = nn.relu(selected_hidden_states) # PriViT R, during inference uncomment this line to replace remainder gelu with relu
        
        updated_hidden_states = hidden_states.at[:, self.true_indices[0], :].set(activated_hidden_states) # tokenwise
        # updated_hidden_states = hidden_states.at[:, self.true_indices[0], self.true_indices[1]].set(activated_hidden_states) # activation wise
        
        return updated_hidden_states

class CustomFlaxViTIntermediate(nn.Module):
    layer: int
    config: ViTConfig
    
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    def setup(self):
        self.dense = nn.Dense(
            self.config.intermediate_size,
            kernel_init=jax.nn.initializers.variance_scaling(
                self.config.initializer_range**2, "fan_in", "truncated_normal"
            ),
            dtype=self.dtype,
        )
        self.patches = int((self.config.image_size*self.config.image_size)/(self.config.patch_size*self.config.patch_size)+1)
        
        self.chooseActivation = ChooseActivation(self.config,self.layer)
        
    def __call__(self, hidden_states):
        
        hidden_states = self.dense(hidden_states)
        hidden_states = self.chooseActivation(hidden_states)
        return hidden_states

class CustomFlaxViTLayer(nn.Module):
    layer: int
    config: ViTConfig
    dtype: jnp.dtype = jnp.float32  

    def setup(self):
        self.attention = CustomFlaxViTAttention(self.layer,self.config, dtype=self.dtype)
        self.intermediate = CustomFlaxViTIntermediate(self.layer,self.config, dtype=self.dtype)
        self.output = FlaxViTOutput(self.config, dtype=self.dtype)
        self.layernorm_before = nn.LayerNorm(epsilon=self.config.layer_norm_eps, dtype=self.dtype)
        self.layernorm_after = nn.LayerNorm(epsilon=self.config.layer_norm_eps, dtype=self.dtype)

    def __call__(self, hidden_states, deterministic: bool = True, output_attentions: bool = False):

        attention_outputs = self.attention(
            self.layernorm_before(hidden_states),  
            deterministic=deterministic,
            output_attentions=output_attentions,
        )

        attention_output = attention_outputs[0]

  
        attention_output = attention_output + hidden_states


        layer_output = self.layernorm_after(attention_output)

        hidden_states = self.intermediate(layer_output)
        hidden_states = self.output(hidden_states, attention_output, deterministic=deterministic)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attention_outputs[1],)
        return outputs


class CustomFlaxViTLayerCollection(FlaxViTLayerCollection):
    config: ViTConfig
    dtype: jnp.dtype = jnp.float32  

    def setup(self):
        self.layers = [
            CustomFlaxViTLayer(i,self.config, name=str(i), dtype=self.dtype) for i in range(self.config.num_hidden_layers)
        ]

    def __call__(self, hidden_states, deterministic: bool = True, output_attentions: bool = False, output_hidden_states: bool = False, return_dict: bool = True,):
        all_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None

        for i in range(self.config.num_hidden_layers):
            layer = self.layers[i]
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_outputs = layer(hidden_states, deterministic=deterministic, output_attentions=output_attentions)

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions += (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        outputs = (hidden_states,)
        if not return_dict:
            return tuple(v for v in outputs if v is not None)

        return FlaxBaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=all_hidden_states, attentions=all_attentions
        )


class CustomFlaxViTEncoder(FlaxViTEncoder):
    config: ViTConfig
    dtype: jnp.dtype = jnp.float32 

    def setup(self):
        self.layer = CustomFlaxViTLayerCollection(self.config, dtype=self.dtype)

    def __call__(
        self,
        hidden_states,
        deterministic: bool = True,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        return super().__call__(hidden_states,
        deterministic = deterministic,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict)






class CustomFlaxViTModule(FlaxViTModule):
    config: ViTConfig
    dtype: jnp.dtype = jnp.float32 
    add_pooling_layer: bool = True

    def setup(self):
        self.embeddings = FlaxViTEmbeddings(self.config, dtype=self.dtype)
        self.encoder = CustomFlaxViTEncoder(self.config, dtype=self.dtype)
        self.layernorm = nn.LayerNorm(epsilon=self.config.layer_norm_eps, dtype=self.dtype)
        self.pooler = FlaxViTPooler(self.config, dtype=self.dtype) if self.add_pooling_layer else None

    def __call__(
        self,
        pixel_values,
        deterministic: bool = True,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):  
        hidden_states = self.embeddings(pixel_values, deterministic=deterministic)
        return super().__call__(pixel_values,
        deterministic=deterministic,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict)






class CustomFlaxViTForImageClassificationModule(FlaxViTForImageClassificationModule):
    config: ViTConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.vit = CustomFlaxViTModule(config=self.config, dtype=self.dtype, add_pooling_layer=False)
        self.classifier = nn.Dense(
            self.config.num_labels,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.variance_scaling(
                self.config.initializer_range**2, "fan_in", "truncated_normal"
            ),
        )

    def __call__(
        self,
        pixel_values=None,
        deterministic: bool = True,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        params = None
    ):

        return super().__call__(pixel_values=pixel_values, deterministic=deterministic, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)


class CustomFlaxViTForImageClassification(FlaxViTPreTrainedModel):
    module_class = CustomFlaxViTForImageClassificationModule


