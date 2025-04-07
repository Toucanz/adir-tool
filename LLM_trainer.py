from datasets import load_dataset, concatenate_datasets
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq, get_scheduler
from peft import LoraConfig, get_peft_model, TaskType
import re
import random
import numpy as np
import nltk
import evaluate
from bert_score import score
import os

############# Control Panel: ##############
dataprep = True
loadModel= True
tokenize = True
trainerSetup = True
train = True
seeValidation = True
add_to_file = True
###########################################

###########################################
############# Data Proccessing ############
###########################################

# FLAN-T5 already has prior knowledge that "TASK:" introduces a directive and "QUESTION:" introduces context.    
def add_tags(example):
    words_to_replace = ("she", "he", "his", "her", "she's", "he's", "he'd", "she'd", "she'll", "he'll")
    if type(example) != str: input_text = example['input_text']
    else: input_text = example

    input_text = input_text.replace("’", "'")   # Fix stupid apostrophes

    ### Tokenizing the text while keeping punctuation
    ### This regex will match words and punctuation
    pattern = r"[^\s.,!?;:]+|[.,!?;:\s]"
    input_tokens = re.findall(pattern, input_text)

    #### Rebuild the input and output texts
    input_text_out = ''.join([f"[CHILD]{word}[/CHILD]" if word.lower() in words_to_replace else word for word in input_tokens])

    if type(example) != str: example['input_text'] = input_text_out
    else: example = input_text_out

    return example

def add_instructions(example):
    inst = "Topic: Eye contact"
    instruction = inst.replace("’", "'")   # Fix stupid apostrophes, again
    if type(example) != str: 
        response = f"#Person1#: {example['input_text']}"
        full_input = f"{instruction}\n{response}"
        example['input_text'] = full_input
    else:
        response = f"RESPONSE: {example}"
        full_input = f"{instruction}\n\n{response}"
        example = full_input
    return example

def CoT_instructions(example):
    inst1 = "[TASK]: The [TRANSCRIPT] is a caretaker describing their child's behavior. Summarize the [TRANSCRIPT] by following these steps, think through each step out loud: Step 1: Summarize each sentence one at a time. Step 2: If the summarized sentence is unrelated to eye contact or attention, remove it. Step 3: Concatenate the remaining sentence summaries together.\n\n"
    inst2 = "[PROCEDURE]:\n"
    inst3 = "STEP 1: Summarize each sentence, with a clinical and objective tone.\n"
    inst4 = "STEP 2: Summarize the summaries from STEP 1. Start each summary with 'The caregiver reported'\n\n"
    if type(example) != str:
        inst5 = f'[TRANSCRIPT]: "{example['input_text']}"'
        full_input = f'{inst1}{inst5}'
        example['input_text'] = full_input
    else:
        inst5 = f'[TRANSCRIPT]: "{example}"'
        full_input = f'{inst1}{inst5}'
        example = full_input      
    return example

def set_CoT_data(CoT_instr, add_tag):
    dataset_CoT = load_dataset('csv', data_files='all_data/qData_CoT.csv')['train']
    for col in ["question_num", "item_num"]:
        if col in dataset_CoT.column_names:
            dataset_CoT = dataset_CoT.remove_columns(col)
    if CoT_instr:
        train_data = dataset_CoT.map(CoT_instructions, batched=False)  # No child tags yet...
    if add_tag:
        train_data = dataset_CoT.map(add_tags, batched=False)
    return train_data, None

def set_curriculum_order():
    # Load datasets in curriculum order
    dataset_easy = load_dataset('csv', data_files='all_data/qDataEasy.csv')['train']
    dataset_medium = load_dataset('csv', data_files='all_data/qDataMedium.csv')['train']
    dataset_hard_full = load_dataset('csv', data_files='all_data/qData.csv')['train']

    # Split only the hard dataset for evaluation
    dataset_hard = dataset_hard_full.train_test_split(test_size=0.2, seed=42)
    train_hard = dataset_hard['train']
    eval_data = dataset_hard['test']

    # Combine all training data in order: easy → medium → hard
    train_data = concatenate_datasets([dataset_easy, dataset_medium, train_hard])
    
    # Remove unnecessary columns if present
    for col in ["question_num", "item_num"]:
        if col in train_data.column_names:
            train_data = train_data.remove_columns(col)
        if col in eval_data.column_names:
            eval_data = eval_data.remove_columns(col)

    train_data = train_data.map(add_tags, batched=False).map(add_instructions, batched=False)   # Add tags and instructions to train data
    eval_data = eval_data.map(add_instructions, batched=False)                                  # Add onlt intructions to validation data

    return train_data, eval_data

LLM_performance_headers = True  # ignore, for LLM_performance.csv
def run_model(training_args, LoRA_params, train_data, eval_data, model_type, log_file=True, show_predictions=False, load_path=None, save_path="./flan-t5-lora-adapter"):
    ###########################################
    ########### Setting Model Params ##########
    ###########################################
    # training_args =  learning rate, no. epochs, label smoothing, max grad norm, train batch size, eval batch size
    lr = training_args[0]
    no_epochs = training_args[1]
    labels_smoothing = training_args[2]
    max_grad_norm = training_args[3]
    train_batch_size = training_args[4]
    eval_batch_size = training_args[5]
    warmup_ratio = training_args[6]
    decay_schedule = training_args[7]
    weight_decay = training_args[8]

    # LoRA_params =  r, alpha, dropout
    if LoRA_params is False: lora_wrap = False
    else:
        r = LoRA_params[0]
        alpha = LoRA_params[1]
        dropout = LoRA_params[2]
        lora_wrap = True
    

    train_data = train_data
    if eval_data == None: eval_data = train_data
    else: eval_data = eval_data


    ###########################################
    ########## Pre-trained model setup ########
    ###########################################

    if loadModel:
        model_name = model_type

        # Loads the tokenizer for the speicifSd model 
        tokenizer = AutoTokenizer.from_pretrained(load_path if load_path else model_name)

        # Loads the pre-trained model
        model = AutoModelForSeq2SeqLM.from_pretrained(load_path if load_path else model_name)

        ### Apply LoRA making training faster and lighter
        if lora_wrap:
            lora_config = LoraConfig(
                task_type=TaskType.SEQ_2_SEQ_LM,  # Sequence-to-sequence task
                r=r,                             # Rank dimension (smaller = less compute)
                lora_alpha=alpha,                    # Scaling factor
                lora_dropout=dropout,                 # Dropout rate (helps generalization)
            )
            # Wrap your model in LoRA
            model = get_peft_model(model, lora_config)


        


    ###########################################
    ########## Tokenize the Dataset ###########
    ###########################################

    # (model can't read raw text, needs to be converted into token IDs the model undersands)
    def preprocess_function(examples, max_input_token, max_output_token):
        # Tokenize inputs
        model_inputs = tokenizer(
            examples['input_text'],
            max_length=512,   #max_input_token,
            padding="max_length",   
            truncation=True
            
        )

        # Tokenize labels (output)
        labels = tokenizer(
            examples['output_text'],
            max_length=256,  #max_output_token,
            padding="max_length",   
            truncation=True,
        )

        # Replace padding token id's in labels by -100 so they are ignored in loss computation
        labels["input_ids"] = [
            [(label if label != tokenizer.pad_token_id else -100) for label in label_seq]
            for label_seq in labels["input_ids"]
        ]
        
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    if tokenize:
        max_input_token_train = max(len(tokenizer(example['input_text'])['input_ids']) for example in train_data)
        max_output_token_train = max(len(tokenizer(example['output_text'])['input_ids']) for example in train_data)
        tokenized_train = train_data.map(lambda examples: preprocess_function(examples, max_input_token_train, max_output_token_train), batched=True)

        if eval_data is None: tokenized_eval = None
        else:
            max_input_token_eval = max(len(tokenizer(example['input_text'])['input_ids']) for example in eval_data)
            max_output_token_eval = max(len(tokenizer(example['output_text'])['input_ids']) for example in eval_data)
            tokenized_eval = eval_data.map(lambda examples: preprocess_function(examples, max_input_token_eval, max_output_token_eval), batched=True)



    ###########################################
    ############# Setup Trainer ###############
    ###########################################
    ## Seq2SeqTrainer automates the training loop, including batching, optimization, logging, etc.

    if trainerSetup:
        nltk.download("punkt_tab", quiet=True)
        metric = evaluate.load("rouge")

        ### Compute ROUGE metrics for evaluation
        def compute_metrics(eval_preds):  
            preds, labels = eval_preds

            # Replace invalid token IDs in predictions
            preds = np.where(preds < 0, tokenizer.pad_token_id, preds)

            # decode preds and labels
            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
            decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
            decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

            # rougeLSum expects newline after each sentence
            decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
            decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]

            rouge_result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)

            # Compute BERTScore
            P, R, F1 = score(decoded_preds, decoded_labels, lang="en", verbose=False)
            bertscore_result = {
            "bertscore_precision": P.mean().item(),
            "bertscore_recall": R.mean().item(),
            "bertscore_f1": F1.mean().item(),
            }
            
            # Combine ROUGE and BERTScore results
            result = {**rouge_result, **bertscore_result}
            return result


        data_collator = DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            model=model,
            label_pad_token_id=-100,
        )

        training_args = Seq2SeqTrainingArguments(
            output_dir="./flan-t5-finetuned",          # Where to save the model
            per_device_train_batch_size=train_batch_size,             # Small batch sizes for low VRAM
            per_device_eval_batch_size=eval_batch_size,
            predict_with_generate=True,                # Enable text generation for validation
            evaluation_strategy="epoch",               # Evaluate after each epoch
            save_strategy="epoch",                     # Save after each epoch
            learning_rate=lr,                          # Learning rate for optimizer (default 1e-4)
            num_train_epochs=no_epochs,                        # Adjust depending on overfitting
            logging_dir="./logs",                      # Logs for TensorBoard
            fp16=False,                                  # Use FP16 if GPU supports it (reduces memory). (causes eval_loss to be NaN on my computer)
            label_smoothing_factor=labels_smoothing,                # Regularization for better generalization
            max_grad_norm=max_grad_norm,                          # gradient clipping
            weight_decay=weight_decay,                          # Regularization for better generalization
            warmup_ratio=warmup_ratio,
            lr_scheduler_type=decay_schedule,
            save_total_limit=1,                   # Limit the total amount of checkpoints. Deletes the older checkpoints.  Only keep the last 1 (best one)
        )

        ## Create Trainer, pass model, data, tokenizer
        trainer = Seq2SeqTrainer(
            model=model,                                                                # The model to be finetuned (FLAN-T5 with LoRA)
            args=training_args,                                                          # Training arguments previously defined
            train_dataset=tokenized_train,                                                  # Tokenized training data
            eval_dataset=tokenized_eval if tokenized_eval is not None else tokenized_train,            # Tokenized validation data
            tokenizer=tokenizer,                                                        # Tokenizer for the model
            data_collator=data_collator,                                                 # Data collator for batching and padding
            compute_metrics=compute_metrics if tokenized_eval is not None else None       # Optional: metrics to evaluate model performance              
        )


    ###########################################
    ############ Train the Model! #############
    ###########################################
    
    if train:
        trainer.train()

        if not os.path.exists(save_path):
            os.makedirs(save_path)
        
        # Save the model
        if lora_wrap:
            model.save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)
        else: 
            model.save_pretrained("./flan-t5-finetuned")
            tokenizer.save_pretrained("./flan-t5-finetuned")




    ###########################################
    ########### Validate the Model ############
    ###########################################

    if seeValidation and eval_data is not None:
        eval_results = trainer.evaluate()  # Optional: eval loss

        # Collect validation inputs (adjust as needed)
        eval_inputs_in = []
        for i in range(3):
            eval_inputs_in.append(eval_data[random.randint(0, len(eval_data)-1)]['input_text'])
        eval_inputs = eval_inputs_in

        # Tokenize the inputs
        inputs = tokenizer(
            eval_inputs,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(model.device)

        # Generate predictions
        generated_tokens = model.generate(
            **inputs,
            min_length=8,
            max_length=256,
            num_beams=8,        # changes "beam search" process, which is a decoding strategy.  Higher values = better quality generation (4 is medium)
            early_stopping=True # optional: stops generation when EOS is reached
        )

        # Decode generated summaries
        decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)



    ###########################################
    ########### Log Performance to File #######
    ###########################################

    if add_to_file and log_file and eval_data is not None:
        # Print input, predicted summary, and reference summary side by side
        global LLM_performance_headers
        prediction_str = ""

        # Column headers for LLM_performance.csv
        if LLM_performance_headers:
            LLM_performance_headers = False
            prediction_str = f"Model Name,eval_loss,ROUGE LSum,ROUGE (1 / 2 / L / Lsum),BERTscore precision,BERTScore Recall,BERTScore F1,Learning Rate,Warmup Ratio,Decay Schedule,No. Epochs,LoRA parameters (r/alpha/dropout),"
            for idx in range(len(decoded_preds)):
                prediction_str += f"Input {idx+1},Predicted Summary {idx+1},Reference Summary {idx+1},"
            prediction_str += "\n"

        # Model parameters fitted into csv string

        prediction_str += f'"{model_name}","{eval_results['eval_loss']}","{eval_results['eval_rougeLsum']}","{eval_results['eval_rouge1']} / {eval_results['eval_rouge2']} / {eval_results['eval_rougeL']} / {eval_results['eval_rougeLsum']}","{eval_results['eval_bertscore_precision']}","{eval_results['eval_bertscore_recall']}","{eval_results['eval_bertscore_f1']}","{lr}","{warmup_ratio}","{decay_schedule}","{no_epochs}","{r}/{alpha}/{dropout}",'
        
        # 3 Decoded predictions fitted into csv string
        for idx in range(len(decoded_preds)):
            try: prediction_str += f'"{eval_inputs[idx].split("#Person1#:")[1]}","{decoded_preds[idx]}","{eval_data[idx]["output_text"]}",'.strip()
            except: prediction_str += f'"{eval_inputs[idx]}","{decoded_preds[idx]}","{eval_data[idx]["output_text"]}",'.strip()
        with open("all_data/LLM_performance.csv", "a") as f:
            f.write(f"{prediction_str}\n")

    if show_predictions and eval_data is not None:
        for idx in range(len(decoded_preds)):
            print(f'Input {idx+1}: "{eval_inputs[idx].split("RESPONSE:")[1]}"\n')
            print(f'Predicted Summary {idx+1}: "{decoded_preds[idx]}"\n')
            print(f'Reference Summary {idx+1}: "{eval_data[idx]["output_text"]}"\n\n')

