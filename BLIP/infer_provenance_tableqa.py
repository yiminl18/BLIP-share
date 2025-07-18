import provenance
import base_strategies
import os, re
current_file_directory = os.path.dirname(os.path.abspath(__file__))
parent_directory = os.path.dirname(current_file_directory)

sufficient_provenance_strategy_pool = ['embedding_sufficient_top_down','embedding_sufficient_bottem_up'] 
minimal_provenance_strategy_pool = ['exponential_greedy'] 

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

def get_table_data(table_name, db_name, table_folder):
    path = table_folder + '64k_' + db_name + '_' + table_name + '.json'
    return read_json(path)

def get_table_text(table_names, sql_q, db_name, table_folder = parent_directory + '/data/'):
    target_table = ''
    for table in table_names:
        if table in sql_q:
            target_table = table
            break 
    if target_table == '':
        print('table not found')
        return 
    table_data = get_table_data(target_table, db_name, table_folder)
    concatenated_table_data = flatten_table(table_data)
    return concatenated_table_data

def provenance_tableqa(data_file, result_folder, embedding_folder, prune_strategy, refine_strategy, model_name, dataset, db_name):
    create_folder_if_not_exists(result_folder)

    data = read_json(data_file)
    provenance.set_model(model_name)
    instruction = 'This is a table-based question. Return only the answer found in the table cell(s); do not include any explanation. The context provides each tuple as a list of column–value pairs. If answer is not found, return NULL.'
    embedding_path = embedding_folder + '/' + dataset + '.npy' 

    i = 0
    for obj in data:
        type = obj['template']
        queries = obj['content']
        new_obj = {}
        new_obj['template'] = type 
        
        for query in queries:
            i += 1
            #print('query id:', i)
            nl = query['nl']
            sql = query['query']
            tables = extract_table_names(sql)
            text = get_table_text(tables, sql, db_name)
            result_path = result_folder + type + '_' + str(i) + '.json' 
            provenance.logger(text, (nl, instruction), '', model_name, result_path, prune_strategy, refine_strategy, metric = 'LLM', embedding_path=embedding_path, data_type = 'table')

if __name__ == "__main__":
    model_name = 'gpt_4o_mini_azure'# gemini2flash
    data_folder = ''
    embedding_folder = ''
    result_folder = ''
    dataset = ['restaurant', 'movie'] 
    db_name = ''

    for data in dataset: 
        if data == 'restaurant':
            data_folder = parent_directory + '/data/restaurant_queries.json'
            db_name = 'restaurant'
        elif data == 'movie':
            data_folder = parent_directory + '/data/movie_queries.json'
            db_name = 'movie'
        
        embedding_folder = parent_directory + '/out/' + data 
        create_folder_if_not_exists(embedding_folder)
        result_folder = parent_directory + '/result/' + data + '_' + model_name + '/'  
        create_folder_if_not_exists(result_folder)
        
        for prune_strategy in sufficient_provenance_strategy_pool:
            for refine_strategy in minimal_provenance_strategy_pool:
                provenance_tableqa(data_folder, result_folder, embedding_folder, prune_strategy, refine_strategy, model_name, data, db_name)



        