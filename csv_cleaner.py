
file_path = "all_data/qDataMedium - Sheet1 (3).csv"

with open(f"{file_path}", "r") as f:

    for c in f.readlines()[0:10]:
        input = c.split(",")[2]
        input_lst = input.split(". ")
        gender= "nothing"
        if "they" in input_lst[0].lower(): gender = "they"
        elif "she" in input_lst[0].lower(): gender = "she"
        elif "he" in input_lst[0].lower(): gender = "he"

        new_sentence = ""
        for sentence in input_lst[1:]:
            for word in sentence.split(" "):
                if word.lower() in ["She", "she", "He", "he", "they", "They"]: 
                    if word.lower() != gender: new_sentence += gender + " "
                    else: new_sentence += word + " "
                else: new_sentence += word + " "
        final_sentence = input_lst[0] + ". " + new_sentence.strip()
        print(f"Gender:{gender}\nOld:{input}\nNew:{final_sentence}\n\n")


        