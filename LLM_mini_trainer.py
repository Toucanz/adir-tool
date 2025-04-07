from LLM_trainer import run_model, add_tags, add_instructions, set_CoT_data
from datasets import load_dataset, concatenate_datasets
import re



def data_column_clean(dataset_name, input_name, output_name, orgin=None):
    #Function takes a dataset and renames the input and output columns to "input_text" and "output_text", and removes any other columns.
    dataset_name = dataset_name.map(lambda x: {"input_text": x[input_name], "output_text": x[output_name]})

    if orgin == "dialogsum":
        dataset_name = dataset_name.map(lambda x: {"input_text": f'Topic: {x["topic"]}\n\n{x["input_text"]}'})

    # Remove unnecessary columns if present
    dataset_name = dataset_name.remove_columns([col for col in dataset_name.column_names if col not in ["input_text", "output_text"]])

    return dataset_name

##### Train on Easy Dataset #####
easy_train = False
if easy_train:
    dataset_easy_full = load_dataset('csv', data_files='all_data/qDataEasy2.csv')['train']
    dataset_easy_full = dataset_easy_full.train_test_split(test_size=0.1, seed=42)
    train_data = data_column_clean(dataset_easy_full['train'], "input_text", "output_text")
    eval_data = data_column_clean(dataset_easy_full['test'], "input_text", "output_text")

    train_data = train_data.map(add_instructions, batched=False)   
    eval_data = eval_data.map(add_instructions, batched=False)                                 


##### Easy+Medium CT #####
easy_medium_train = False
if easy_medium_train:
    dataset_easy = load_dataset('csv', data_files='all_data/qDataEasy2.csv')['train']
    dataset_medium = load_dataset('csv', data_files='all_data/qDataMedium2.csv')['train']
    
    full_data = concatenate_datasets([dataset_easy, dataset_medium]).train_test_split(test_size=0.12, seed=42)

    
    train_data = data_column_clean(full_data['train'], "input_text", "output_text")
    eval_data = data_column_clean(full_data['test'], "input_text", "output_text")

    train_data = train_data.map(add_instructions, batched=False)   
    eval_data = eval_data.map(add_instructions, batched=False)   


##### Easy+Medium NO CT #####
easy_medium_train = True
if easy_medium_train:
    dataset_medium = load_dataset('csv', data_files='all_data/qDataMedium2.csv')['train']
    
    full_data = dataset_medium.train_test_split(test_size=0.1, seed=42)

    
    train_data = data_column_clean(full_data['train'], "input_text", "output_text")
    eval_data = data_column_clean(full_data['test'], "input_text", "output_text")

    train_data = train_data.map(add_instructions, batched=False)   
    eval_data = eval_data.map(add_instructions, batched=False)  


##### CoT Dataset #####
#train_data, eval_data = set_CoT_data(True, True)  # eval_data is None
#train_data, eval_data = set_curriculum_order()


##### dialogsum Dataset #####
dialogsum_train = False
if dialogsum_train:
    ds = load_dataset("knkarthick/dialogsum")
    train_data = ds['train']
    eval_data = ds['validation']
    train_data = data_column_clean(train_data, "dialogsum", "dialogue", "summary")
    eval_data = data_column_clean(eval_data, "dialogsum", "dialogue", "summary")


LoRA_args = [2, 2, 0.1]  # r, alpha, dropout
training_args = [1e-3, 6, 0.1, 0, 4, 4, 0.05, "cosine", 0.0]  # learning rate, no. epochs, label smoothing, max grad norm, train batch size, eval batch size, warmup ratio, decay schedule, weight decay
model_type = "google/flan-t5-base"

run_model_bool = True

if run_model_bool:
    run_model(
            [3e-4, 6, 0.1, 0, 8, 8, 0.1, "cosine", 0.01], 
            LoRA_args, 
            train_data, 
            eval_data, 
            model_type, 
            log_file=True,
            load_path="./flan-t5-lora-adapter-dialougeSum-easy",
            save_path="./flan-t5-lora-adapter-dialougeSum_easyMedium",  
                )