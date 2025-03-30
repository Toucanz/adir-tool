from datasets import load_dataset, concatenate_datasets
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq, T5Tokenizer, T5ForConditionalGeneration
from peft import LoraConfig, get_peft_model, TaskType
import re
import random
import numpy as np
import nltk
import evaluate
from bert_score import score

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

LLM_performance_headers = True  # ignore, for LLM_performance.csv
def run_model(training_args, LoRA_params, data_args):
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

    # LoRA_params =  r, alpha, dropout
    r = LoRA_params[0]
    alpha = LoRA_params[1]
    dropout = LoRA_params[2]

    # data_args = instruction, tags
    inst = data_args[0]
    tags = data_args[1]
    
    addInstructions = True
    if inst == None:
        addInstructions = False
    
    ###########################################
    ############### Prepare Data ##############
    ###########################################

    if dataprep:
        # FLAN-T5 already has prior knowledge that "TASK:" introduces a directive and "QUESTION:" introduces context.    
        def add_tags(example):
            words_to_replace = ("she", "he", "his", "her", "she's", "he's", "he'd", "she'd", "she'll", "he'll")
            input_text = example['input_text']
            #output_text = example['output_text']

            input_text = input_text.replace("’", "'")   # Fix stupid apostrophes

            ### Tokenizing the text while keeping punctuation
            ### This regex will match words and punctuation
            pattern = r"[^\s.,!?;:]+|[.,!?;:\s]"
            input_tokens = re.findall(pattern, input_text)
            #output_tokens = re.findall(pattern, output_text)

            #### Rebuild the input and output texts
            input_text_out = ''.join([f"[CHILD]{word}[/CHILD]" if word.lower() in words_to_replace else word for word in input_tokens])
            #output_text_out = ''.join([f"[NAME]{word}[/NAME]" if word.lower() in words_to_replace else word for word in output_tokens])

            example['input_text'] = input_text_out
            #example['output_text'] = output_text_out
            return example
        
        def add_instructions(example):
            instruction = inst.replace("’", "'")   # Fix stupid apostrophes, again
            response = f"RESPONSE: {example['input_text']}"
            full_input = f"{instruction}\n\n{response}"
            example['input_text'] = full_input
            return example


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

        if tags:
            train_data = train_data.map(add_tags, batched=False)
        
        if addInstructions:
            train_data = train_data.map(add_instructions, batched=False)
            eval_data = eval_data.map(add_instructions, batched=False)  # No tags added for validation data



    ###########################################
    ########## Pre-trained model setup ########
    ###########################################

    if loadModel:
        model_name = "google/flan-t5-base"

        # Loads the tokenizer for the speicifed model 
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Loads the pre-trained model
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

        ### Apply LoRA making training faster and lighter
        lora_config = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,  # Sequence-to-sequence task
            r=r,                             # Rank dimension (smaller = less compute)
            lora_alpha=alpha,                    # Scaling factor
            lora_dropout=dropout                 # Dropout rate (helps generalization)
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
            max_length=max_input_token,
            padding="max_length",   
            truncation=True
        )

        # Tokenize labels (output)
        labels = tokenizer(
            examples['output_text'],
            max_length=max_output_token,
            padding="max_length",   
            truncation=True
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
            weight_decay=0.01,                          # Regularization for better generalization
        )

        ## Create Trainer, pass model, data, tokenizer
        trainer = Seq2SeqTrainer(
            model=model,                              # The model to be finetuned (FLAN-T5 with LoRA)
            args=training_args,                       # Training arguments previously defined
            train_dataset=tokenized_train,            # Tokenized training data
            eval_dataset=tokenized_eval,              # Tokenized validation data
            tokenizer=tokenizer,                      # Tokenizer for the model
            data_collator=data_collator,               # Data collator for batching and padding
            compute_metrics=compute_metrics                      # Optional: metrics to evaluate model performance              
        )



    ###########################################
    ############ Train the Model! #############
    ###########################################
    
    if train:
        trainer.train()

        # Save the model
        model.save_pretrained("./flan-t5-lora-adapter")
        tokenizer.save_pretrained("./flan-t5-lora-adapter")



    ###########################################
    ########### Validate the Model ############
    ###########################################

    if seeValidation:
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
            min_length=24,
            max_length=256,
            num_beams=8,        # changes "beam search" process, which is a decoding strategy.  Higher values = better quality generation (4 is medium)
            early_stopping=True # optional: stops generation when EOS is reached
        )

        # Decode generated summaries
        decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)



    ###########################################
    ########### Log Performance to File #######
    ###########################################

    if add_to_file:
        # Print input, predicted summary, and reference summary side by side
        global LLM_performance_headers
        prediction_str = ""
        if LLM_performance_headers:
            LLM_performance_headers = False
            prediction_str = f"Model Name,eval_loss,ROUGE LSum,ROUGE (1 / 2 / L / Lsum),BERTscore precision,BERTScore Recall,BERTScore F1,tags,instruction,Learning Rate,No. Epochs,LoRA parameters (r/alpha/dropout),"
            for idx in range(len(decoded_preds)):
                prediction_str += f"Input {idx+1},Predicted Summary {idx+1},Reference Summary {idx+1},"
            prediction_str += "\n"

        prediction_str += f'"{model_name}","{eval_results['eval_loss']}","{eval_results['eval_rougeLsum']}","{eval_results['eval_rouge1']} / {eval_results['eval_rouge2']} / {eval_results['eval_rougeL']} / {eval_results['eval_rougeLsum']}","{eval_results['eval_bertscore_precision']}","{eval_results['eval_bertscore_recall']}","{eval_results['eval_bertscore_f1']}","{tags}","{inst}","{lr}","{no_epochs}","{r}/{alpha}/{dropout}",'
        #print(f"Validation Loss: {eval_results}")
        for idx in range(len(decoded_preds)):
            
            #print(f"Input {idx+1}: {eval_inputs[idx]}")
            #print(f"Predicted Summary {idx+1}: {decoded_preds[idx]}")
            #print(f"Reference Summary {idx+1}: {eval_data[idx]['output_text']}\n")
            if inst != None: prediction_str += f'"{eval_inputs[idx].split("RESPONSE:")[1]}","{decoded_preds[idx]}","{eval_data[idx]['output_text']}",'.strip()
            else: prediction_str += f'"{eval_inputs[idx]}","{decoded_preds[idx]}","{eval_data[idx]["output_text"]}",'.strip()
        
        with open("all_data/LLM_performance.csv", "a") as f:
            f.write(f"{prediction_str}\n")



###########################################
################# Parameters ##############
###########################################    

learning_rates_list =  [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1]
instruction_list = [
    "TASK: You are a clinical interviewer. Summarize the following text reporting the child's eye contact behaviour",
    "TASK: You are a clinical interviewer. Summarize the following text",
    "TASK: As a clinical interviewer, summarize the eye contact behaviour described below",
    "TASK: Generate a clinician-style summary describing the child's eye contact behaviour",
    "TASK: Write a clinician-style summary using the parent's description of the child's eye contact",
    "TASK: Recast this parent narrative into a clinician's objective report on eye contact",
    "TASK: Produce a diagnostic-style description of the child's eye contact based on the text",
    "TASK: Summarize the following text reporting the child's eye contact behaviour",
    "TASK: Summarize the child's social gaze behaviour as described by the parent",
    "TASK: Summarize the child's social gaze based on the following parent comments",
    "TASK: Provide a formal summary of the child's behavior related to eye contact",
    "TASK: Based on the parent's account, summarize how the child used eye contact",
    "TASK: Compose a professional summary of the child's visual attention during interactions",
    "TASK: Describe the child's responsiveness to visual social cues based on the parent's report",
    "TASK: Summarize how the child initiates and maintains eye contact in social situations",
    "TASK: Write a clinical summary of the child's eye contact behaviour based on the following report",
    "TASK: Based on the parent's statement, describe the child's eye contact patterns in a clinical tone",
    "TASK: Given the following parent narrative, write a clinical summary of eye contact behaviour",
    "TASK: Transform the parent's informal account into a clinical summary of gaze behaviour",
    "TASK: Turn this informal parent description into a formal clinical observation of eye contact",
    "TASK: Convert the following into a clinical observation of eye contact behaviour"
]
instruction_list_4 = [
    "TASK: Write a professional behavioral summary describing eye contact between ages 4 and 5",
    "TASK: Write a summary of how the child engages visually during social interaction",
    "TASK: Summarize the parent’s observations of the child’s eye contact into a clinical report",
    "TASK: Create a brief, objective summary of the child’s social gaze as reported by the parent",
    "TASK: Write a structured clinical note on the child's eye contact between ages 4 and 5",
    "TASK: Write a concise behavioral summary about the child's eye contact from ages 4 to 5",
    "TASK: Provide a summary suitable for a psychological assessment report about eye contact",
    "TASK: Create an objective summary of the child’s visual engagement with others",
    "TASK: Generate a psychological report summary of the child’s eye contact"
]



LoRA_args = [8, 64, 0.1]  # r, alpha, dropout
training_args = [1e-4, 1, 0.1, 0.5, 4, 4]  # learning rate, no. epochs, label smoothing, max grad norm, train batch size, eval batch size
data_args = [None, True]  # instruction, tags
####  run_model(1e-4, None, 10, [32, 64, 0.0], True)    # default =  learning rate, instruction, no. epochs, LoRA parameters (r, alpha, dropout), tags

#run_model(training_args, LoRA_args, [None, True])
#run_model(training_args, LoRA_args, [None, False])

batch_learn = True
accum1 = 1
if batch_learn:
    for test_item in instruction_list:
        print(f"LLM Test {accum1}/{len(instruction_list)}")
        accum1 += 0.5
        run_model([1e-3, 3, 0.1, 0.5, 4, 4], LoRA_args, [test_item, True])
        accum1 += 0.5
        run_model([1e-3, 5, 0.1, 0.5, 4, 4], LoRA_args, [test_item, True])
