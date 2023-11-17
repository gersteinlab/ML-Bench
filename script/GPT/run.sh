
# choose from quarter set or full set
type="quarter"

# model name
model="4"

input_file="input_file.jsonl"

answer_file=""$model"_"$type".json"

parsing_file=""$model"_"$type".jsonl"

# choose from oracle_segment and readme
# oracle_segment: The code paragraph in the readme that is most relevant to the task
# readme: The entire text of the readme in the repository where the task is located
readme_type="readme"

instructions="extend_instructions"

# choose from gpt-35-turbo-16k and gpt-4-32
engine_name="gpt-4-32k"

n_turn=1

python query_gpt.py \
 --readme_type ${readme_type} \
 --instruction ${instructions} \
 --nturn ${n_turn} \
 --engine ${engine_name} \
 --input_file ${input_file} \
 --answer_file ${answer_file} \
 --parsing_file ${parsing_file} \
 --openai_key xxxx