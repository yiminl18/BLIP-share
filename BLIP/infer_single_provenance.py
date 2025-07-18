import provenance
import base_strategies
import os, re
current_file_directory = os.path.dirname(os.path.abspath(__file__))
parent_directory = os.path.dirname(current_file_directory)

sufficient_provenance_strategy_pool = ['embedding_sufficient_top_down','embedding_sufficient_bottem_up','LLM_score_sufficient_bottem_up','LLM_score_sufficient_top_down','LLM_adaptive','Embedding_adaptive'] 
minimal_provenance_strategy_pool = ['null','exponential_greedy','sequential_greedy'] 

import json
def read_json(path):
    with open(path, "r", encoding="utf-8") as file:
        data = json.load(file)
    return data

def get_sufficient_result(sufficient_path):
    latency = -1
    in_tokens = -1
    out_tokens = -1
    provenance_ids = [-1]
    eval_time = -1
    answer = ['']
    status = 'N/A'
    if os.path.isfile(sufficient_path):
        result = read_json(sufficient_path)
        if 'time' in result and 'tokens' in result and 'provenance_ids' in result and 'eval_time' in result:
            latency = result['time']
            (in_tokens, out_tokens) = result['tokens']
            provenance_ids = result['provenance_ids']
            eval_time = result['eval_time']
        if 'answer' in result:
            answer = result['answer']
        if 'status' in result:
            status = result['status']
    return latency, (in_tokens, out_tokens), provenance_ids, eval_time, answer, status

def write_json_to_file(filename, data):
    with open(filename, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4, ensure_ascii=False)

                
def get_embedding_path(data, embedding_folder, id, object):
    embedding_path = ''
    id = id-1
    if data == 'paper':
        return embedding_folder + '/embeddings/' + 'paper_' + str(id) + '_' + object['id'] + '_embeddings.npy'
    if data == 'nl_dev': 
        return embedding_folder + '/embeddings/' + 'nl_' + str(id) + '_embeddings.npy'
    if data == 'hotpotQA':
        return embedding_folder + '/embeddings/' + 'hotpot' + '_q' + str(id) + '_embeddings.npy'
    

#for gpt4omini
def get_result_path(data, result_folder_path, id, object, strategy, model):
    if data == 'paper':
        return result_folder_path + str(id) + '_' + str(object['id']) + '_'  + strategy  +  '.json'
    if data == 'nl_dev':
        return result_folder_path + str(id) + '_' + str(object['id']) + '_'  + strategy  + '.json'
    if data == 'hotpotQA':
        return result_folder_path + str(id) + '_' + str(object['document_name']) + '_'  + strategy + '.json'

def get_sufficient_path(data, result_folder_path, id, object, sufficient_provenance_strategy, model): 
    if data == 'paper':
        return result_folder_path + str(id) + '_' + str(object['id']) + '_'  + sufficient_provenance_strategy + '_null' + '_' + model + '.json'
    if data == 'nl_dev':
        return result_folder_path + str(id) + '_' + str(object['id']) + '_'  + sufficient_provenance_strategy + '_null' + '_' + model + '.json'
    if data == 'hotpotQA':
        return result_folder_path + str(id) + '_' + str(object['document_name']) + '_'  + sufficient_provenance_strategy + '_null' + '_' + model +  '.json'
    
def get_minimal_result(result_path):
    result = read_json(result_path)
    status = 'N/A' 
    if 'status' in result:
        status = result['status'] 
    return status 


def provenance_run(data, data_path, embedding_folder, result_folder_path, model_name, num_case):
    objects = read_json(data_path)
    instruction = 'Only return answers. Do not add explanations. If answers are not found in the given context, return NULL. Context: '
    i = 0

    for o in objects:
        if data == 'hotpotQA':
            text = o['context']
            title = o['document_name']
        else: 
            text = o['text']
            title = o['id']
        q = o['question']
        question = (q, instruction)
        i += 1

        if i > num_case:
            break

        for sufficient_provenance_strategy in sufficient_provenance_strategy_pool:
            for minimal_provenance_strategy in minimal_provenance_strategy_pool:
                strategy = sufficient_provenance_strategy + '_' + minimal_provenance_strategy

                if sufficient_provenance_strategy != 'LLM_adaptive':
                    continue  
                
                embedding_path = get_embedding_path(data, embedding_folder, i, o)
                result_path = get_result_path(data, result_folder_path, i, o, strategy, model_name)
                sufficient_path = get_sufficient_path(data, result_folder_path, i, o, sufficient_provenance_strategy, model_name)

                sufficient_time, sufficient_tokens, sufficient_provenance_ids, sufficient_eval_latency, sufficient_answers, sufficient_status =  get_sufficient_result(sufficient_path)
                print(i, strategy)
                
                provenance.logger(text, question, title, model_name, result_path, sufficient_provenance_strategy, minimal_provenance_strategy, metric = 'LLM', embedding_path=embedding_path, sufficient_time = sufficient_time, sufficient_tokens = sufficient_tokens, sufficient_provenance_ids = sufficient_provenance_ids, sufficient_eval_latency = sufficient_eval_latency)


def create_folder_if_not_exists(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

def extract_table_names(sql_query):
    """
    Extract all table names from a SQL SELECT query.
    Returns a list of table names (handles FROM and JOIN).
    """
    # Pattern matches FROM or JOIN followed by optional backticks and the table name
    pattern = r'(?:FROM|JOIN)\s+`?([a-zA-Z_][\w]*)`?'
    return re.findall(pattern, sql_query, flags=re.IGNORECASE)

def flatten_table(data):
    """
    Flatten each dict in a list into a string of key=value pairs.
    Prefix each with 'tuple {i}:' where i is the index.
    """
    output = []
    for i, item in enumerate(data):
        parts = [f"{k}={v}" for k, v in item.items()]
        line = f"tuple {i}: " + ", ".join(parts)
        output.append(line)
    return output


if __name__ == "__main__":

    model_name = 'gpt_4o_mini_azure'# gemini2flash
    data_folder = ''
    embedding_folder = ''
    result_folder = ''
    num_case = 500
    dataset = ['paper','hotpotQA','nl_dev'] 

    for data in dataset: 
        if data == 'paper':
            data_folder = parent_directory + '/data/qasper_sample_papers.json'
        elif data == 'nl_dev':
            data_folder = parent_directory + '/data/natural-questions_nq-dev-full.json'
        elif data == 'hotpotQA':
            data_folder = parent_directory + '/data/hotpotQA_fullwiki.json'
        
        embedding_folder = parent_directory + '/out/' + data 
        create_folder_if_not_exists(embedding_folder)
        result_folder = parent_directory + '/result/' + data + '_' + model_name + '/'  
        create_folder_if_not_exists(result_folder)

        provenance_run(data, data_folder, embedding_folder, result_folder, model_name, num_case)

        