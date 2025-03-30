from datasets import load_dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq
from peft import LoraConfig, get_peft_model, TaskType

############# Control Panel: ##############
dataprep = True
loadModel= True
tokenize = True
trainerSetup = True
train = True
seeValidation = True
###########################################

###########################################
############# Data Proccessing ############
###########################################

if dataprep:
    # reads csv into a HuggingFace DatasetDict, allowing effieient processsing and training of data
    dataset = load_dataset('csv', data_files=r'C:\Users\maxim\Desktop\SS LLC\qData.csv')

    # drop unused columns (only need input_text and output_text)
    dataset = dataset.remove_columns(['question_num', 'item_num'])


    # Labels like TASK:, QUESTION:, PARENT RESPONSE: are tokens, and the model learns their meaning within the context of the data. 
    # FLAN-T5 already has prior knowledge that "TASK:" introduces a directive and "QUESTION:" introduces context.
    def add_instruction(example):
        instruction = "TASK: Write a formal summary of the child's eye contact behavior in a clinical tone, based on the following parent statement.  Be very detailed and write at least 2 sentences."
        response = f"RESPONSE: {example['input_text']}"
        full_input = f"{instruction}\n\n{response}"
        example['input_text'] = full_input
        return example
    
    dataset = dataset['train'].train_test_split(test_size=0.1, seed=42)
    train_data = dataset['train'].map(add_instruction, batched=False)
    eval_data = dataset['test'].map(add_instruction, batched=False)



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
        r=64,                             # Rank dimension (smaller = less compute)
        lora_alpha=128,                    # Scaling factor
        lora_dropout=0.0                  # Dropout rate (helps generalization)
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
    print(f"Max input token length (train): {max_input_token_train}")
    print(f"Max output token length (train): {max_output_token_train}")
    tokenized_train = train_data.map(lambda examples: preprocess_function(examples, max_input_token_train, max_output_token_train), batched=True)

    max_input_token_eval = max(len(tokenizer(example['input_text'])['input_ids']) for example in eval_data)
    max_output_token_eval = max(len(tokenizer(example['output_text'])['input_ids']) for example in eval_data)
    print(f"Max input token length (eval): {max_input_token_eval}")
    print(f"Max output token length (eval): {max_output_token_eval}")
    tokenized_eval = eval_data.map(lambda examples: preprocess_function(examples, max_input_token_eval, max_output_token_eval), batched=True)



###########################################
############# Setup Trainer ###############
###########################################
## Seq2SeqTrainer automates the training loop, including batching, optimization, logging, etc.

if trainerSetup:
    
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        label_pad_token_id=-100,
    )

    training_args = Seq2SeqTrainingArguments(
        output_dir="./flan-t5-finetuned",          # Where to save the model
        per_device_train_batch_size=4,             # Small batch sizes for low VRAM
        per_device_eval_batch_size=4,
        predict_with_generate=True,                # Enable text generation for validation
        evaluation_strategy="epoch",               # Evaluate after each epoch
        save_strategy="epoch",                     # Save after each epoch
        learning_rate=1e-4,                        # Learning rate for optimizer
        num_train_epochs=50,                        # Adjust depending on overfitting
        logging_dir="./logs",                      # Logs for TensorBoard
        fp16=True,                                  # Use FP16 if GPU supports it (reduces memory)
        label_smoothing_factor=0.1,                # Regularization for better generalization
    )

    ## Create Trainer, pass model, data, tokenizer
    trainer = Seq2SeqTrainer(
        model=model,                              # The model to be finetuned (FLAN-T5 with LoRA)
        args=training_args,                       # Training arguments previously defined
        train_dataset=tokenized_train,            # Tokenized training data
        eval_dataset=tokenized_eval,              # Tokenized validation data
        tokenizer=tokenizer,                      # Tokenizer for the model
        data_collator=data_collator               # Data collator for batching and padding         
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
    trainer.evaluate()  # Optional: eval loss

    # Collect validation inputs (adjust as needed)
    eval_inputs = eval_data[:5]["input_text"]

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
        max_length=256,
        num_beams=4,        # optional: better quality generation
        early_stopping=True # optional: stops generation when EOS is reached
    )

    # Decode generated summaries
    decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

    # Print input, predicted summary, and reference summary side by side
    for idx in range(len(decoded_preds)):
        print(f"Input {idx+1}: {eval_inputs[idx]}")
        print(f"Predicted Summary {idx+1}: {decoded_preds[idx]}")
        print(f"Reference Summary {idx+1}: {eval_data[idx]['output_text']}\n")

