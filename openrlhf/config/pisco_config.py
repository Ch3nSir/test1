from transformers import PretrainedConfig
class COCOMConfig(PretrainedConfig):

    model_type = "COCOM"
    def __init__(self,
                decoder_model_name: str = "meta-llama/Llama-2-7b-chat-hf",
                doc_max_length: int = 128,
                quantization: str = 'no',
                sep: bool = False,
                compr_model_name: str = "google-bert/bert-base-uncased",
                compr_rate: int = 64,
                compr_n_layers: int = None, # only for surgical mistral compressor
                compr_every_n_layer: int = None,
                compr_base_model_name: str = '/mnt/ceph_rbd/model/Mistral-7B-Instruct-v0.2',
                compr_rms_norm: bool = False, # only for surgical mistral compressor: if true, rms norm applied on h-s
                compr_mlp_hidden_dim: int = 8096,
                compr_use_mlp: bool = True, 
                lora: bool = False, # lora on decoder (and decoder as compr)
                lora_compressor: bool = False, # lora only on the compressor if it exists
                training_form: str = "both",
                training_stage: str = "stage1",
                lora_r: int = 16,
                lora_r_compressor: int = None,
                load_adapters: bool = True,
                kbtc_training: bool = False,
                optimize_mem_tokens: bool = False,
                different_mem_tokens: bool = False,
                attn_implementation: str = 'flash_attention_2',
                device_map = None,
                **kwargs):
        super().__init__(**kwargs)

        self.decoder_model_name = decoder_model_name # model name of decoder
        self.doc_max_length = doc_max_length # the maximum length of document that can be used by this model (it is used to compute number of mem tokens !)
        self.quantization = quantization # quantization, could be no, int4, int8
        self.sep = sep # boolean type, whether to use sep token
        
        self.compr_model_name = compr_model_name # model name of compressor
        self.compr_rate = compr_rate # compression rate
        self.compr_use_mlp = compr_use_mlp
        self.compr_mlp_hidden_dim = compr_mlp_hidden_dim
        self.compr_n_layers = compr_n_layers
        self.compr_every_n_layer = compr_every_n_layer
        self.compr_base_model_name = compr_base_model_name
        self.compr_rms_norm = compr_rms_norm
        
        self.lora = lora # boolean type, whether to use lora trsining
        self.lora_compressor = lora_compressor
        self.training_form = training_form # training form, could be compressor: training only comprssor; both: training both
        self.training_stage = training_stage # training stage, could be stage1 or stage2
        # Or both_separately: training both with separate adapters
        self.lora_r = lora_r # lora_r for lora training, we use 16 throughout the experiment.
        self.lora_r_compressor = lora_r_compressor or lora_r # defaulting to same lora as decoder.
        self.load_adapters = load_adapters # used to load pretrained model: we first load without adapters, and then load them from file.
        self.optimize_mem_tokens = optimize_mem_tokens
        self.different_mem_tokens = different_mem_tokens
        
        self.kbtc_training = kbtc_training
        
        self.device_map = device_map
        
        self.attn_implementation = attn_implementation
        
        if training_form == 'compressor':
            assert compr_model_name is not None and not self.lora
            
