from datasets import load_dataset

def load_data(dataset_name: str = None, dataset_config: str = None):
    """
    Loads the data from either a given name, or wikitext2

    :param str dataset_name: The name of the dataset
    :param str dataset_config: The name of the specific config for the dataset

    :return: the dataset
    """
    if dataset_name is None and dataset_config is None:
        dataset_name = "wikitext"
        dataset_config = "wikitext-2-raw-v1"

    
    dataset = load_dataset(dataset_name, dataset_config) if dataset_config is not None else load_dataset(dataset_name)

    return dataset