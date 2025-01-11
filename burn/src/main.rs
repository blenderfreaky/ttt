use burn::{
    config::Config, module::Module, prelude::Backend, tensor::{Device, Tensor}
};

fn compute<B: Backend>() {
    let device = Default::default();

    let tensor1 = Tensor::<B, 2>::from_floats([[2., 3.], [4., 5.]], &device);
    let tensor2 = Tensor::ones_like(&tensor1);

    let tensor3 = tensor1 + tensor2;

    println!("{:?}", tensor3);
}

fn main() {
    compute::<burn::backend::Wgpu>();
}

/*

class TTTConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`TTTModel`]. It is used to instantiate an TTT
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the TTT-1B.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        vocab_size (`int`, *optional*, defaults to 32000):
            Vocabulary size of the LLaMA model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`LlamaModel`]
        hidden_size (`int`, *optional*, defaults to 4096):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 11008):
            Dimension of the MLP representations.
        num_hidden_layers (`int`, *optional*, defaults to 32):
            Number of hidden layers in the Transformer decoder.
        num_attention_heads (`int`, *optional*, defaults to 32):
            Number of attention heads for each attention layer in the Transformer decoder.
        num_key_value_heads (`int`, *optional*):
            This is the number of key_value heads that should be used to implement Grouped Query Attention. If
            `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if
            `num_key_value_heads=1 the model will use Multi Query Attention (MQA) otherwise GQA is used. When
            converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed
            by meanpooling all the original heads within that group. For more details checkout [this
            paper](https://arxiv.org/pdf/2305.13245.pdf). If it is not specified, will default to
            `num_attention_heads`.
        hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
            The non-linear activation function (function or string) in the decoder.
        max_position_embeddings (`int`, *optional*, defaults to 2048):
            The maximum sequence length that this model might ever be used with. Llama 1 supports up to 2048 tokens,
            Llama 2 up to 4096, CodeLlama up to 16384.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        rms_norm_eps (`float`, *optional*, defaults to 1e-06):
            The epsilon used by the rms normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        pad_token_id (`int`, *optional*):
            Padding token id.
        bos_token_id (`int`, *optional*, defaults to 1):
            Beginning of stream token id.
        eos_token_id (`int`, *optional*, defaults to 2):
            End of stream token id.
        pretraining_tp (`int`, *optional*, defaults to 1):
            Experimental feature. Tensor parallelism rank used during pretraining. Please refer to [this
            document](https://huggingface.co/docs/transformers/main/perf_train_gpu_many#tensor-parallelism) to understand more about it. This value is
            necessary to ensure exact reproducibility of the pretraining results. Please refer to [this
            issue](https://github.com/pytorch/pytorch/issues/76232).
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether to tie weight embeddings
        rope_theta (`float`, *optional*, defaults to 10000.0):
            The base period of the RoPE embeddings.
        rope_scaling (`Dict`, *optional*):
            Dictionary containing the scaling configuration for the RoPE embeddings. Currently supports two scaling
            strategies: linear and dynamic. Their scaling factor must be a float greater than 1. The expected format is
            `{"type": strategy name, "factor": scaling factor}`. When using this flag, don't update
            `max_position_embeddings` to the expected new maximum. See the following thread for more information on how
            these scaling strategies behave:
            https://www.reddit.com/r/LocalLLaMA/comments/14mrgpr/dynamically_scaled_rope_further_increases/. This is an
            experimental feature, subject to breaking API changes in future versions.
        use_gate (`bool`, *optional*, defaults to `False`): whether use gating in Mamba backbone
        share_qk (`bool`, *optional*, defaults to `False`): whether share Q/K projection matrix
        ttt_layer_type (`str`, *optional*, defaults to `"linear"`): ttt block type, "linear" or "mlp", stands for TTT-Linear and TTT-MLP
        ttt_base_lr (`float`, *optional*, defaults to 1.0): base learning rate for TTT learner
        pre_conv (`bool`, *optional*, defaults to `False`): whether use conv before TTT
        conv_kernel (`int`, *optional*, defaults to 4): kernel size of the conv layer
        scan_checkpoint_group_size (`int`, *optional*, defaults to 0):
            gradient checkpoint group size on seq dimension, 0 means no checkpointing.
            In JAX implementation, we set it 4, which means we group 4 mini-batches together in 1 gradient checkpointg to save memory.


    ```python
    >>> from . import TTTModel, TTTConfig

    >>> # Initializing a TTT ttt-1b style configuration
    >>> configuration = TTTConfig()

    >>> # Initializing a model from the ttt-1b style configuration
    >>> model = TTTModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "ttt"

    def __init__(
        self,
        vocab_size=32000,
        hidden_size=2048,
        intermediate_size=5504,
        num_hidden_layers=24,
        num_attention_heads=32,
        hidden_act="silu",
        max_position_embeddings=2048,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=False,
        pad_token_id=None,
        bos_token_id=1,
        eos_token_id=2,
        pretraining_tp=1,
        tie_word_embeddings=True,
        rope_theta=10000.0,
        use_gate=False,
        share_qk=False,
        ttt_layer_type="linear",
        ttt_base_lr=1.0,
        mini_batch_size=16,
        pre_conv=False,
        conv_kernel=4,
        scan_checkpoint_group_size=0,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads

        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.pretraining_tp = pretraining_tp
        self.use_cache = use_cache
        self.rope_theta = rope_theta

        self.use_gate = use_gate
        self.share_qk = share_qk
        self.ttt_layer_type = ttt_layer_type
        self.ttt_base_lr = ttt_base_lr
        self.mini_batch_size = mini_batch_size

        self.pre_conv = pre_conv
        self.conv_kernel = conv_kernel
        self.scan_checkpoint_group_size = scan_checkpoint_group_size

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

*/

#[derive(Config, Debug)]
enum Activation {
    Silu,
}

#[derive(Config, Debug)]
enum TTTLayerType {
    Linear,
    Mlp,
}

#[derive(Config, Debug)]
struct TTTConfig {
    vocab_size: u32,
    hidden_size: u32,
    intermediate_size: u32,
    num_hidden_layers: u32,
    num_attention_heads: u32,
    hidden_act: Activation,
    max_position_embeddings: u32,
    initializer_range: f32,
    rms_norm_eps: f32,
    use_cache: bool,
    pad_token_id: Option<u32>,
    bos_token_id: u32,
    eos_token_id: u32,
    pretraining_tp: i32,
    tie_word_embeddings: bool,
    rope_theta: f32,
    use_gate: bool,
    share_qk: bool,
    ttt_layer_type: TTTLayerType,
    ttt_base_lr: f32,
    mini_batch_size: u32,
    pre_conv: bool,
    conv_kernel: u32,
    scan_checkpoint_group_size: u32,
}