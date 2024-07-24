import yaml
import numpy as np

def get_model_identifiers_from_yaml(model_family):
    # path to model_configs.yaml
    '''
    models:
        llama2-7b:
            hf_key: "NousResearch/Llama-2-7b-chat-hf"
            question_start_tag: "[INST] "
            question_end_tag: " [/INST] "
            answer_tag: ""
            start_of_sequence_token: "<s>"
    '''
    model_configs  = {}
    with open("config/model_config.yaml", "r") as f:
        model_configs = yaml.load(f, Loader=yaml.FullLoader)
    return model_configs[model_family]

def add_dataset_index(dataset):
    indexing = np.arange(len(dataset))
    dataset = dataset.add_column('index', indexing)
    return dataset