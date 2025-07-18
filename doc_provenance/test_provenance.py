import provenance
import data_digestion, base_strategies
import os, re
import concurrent.futures
current_file_directory = os.path.dirname(os.path.abspath(__file__))
parent_directory = os.path.dirname(current_file_directory)

# sufficient_provenance_strategy_pool = ['embedding_sufficient_top_down','embedding_sufficient_bottem_up','LLM_score_sufficient_bottem_up','LLM_score_sufficient_top_down', 'divide_and_conquer_sufficient'] 
# minimal_provenance_strategy_pool = ['null','exponential_greedy','sequential_greedy'] 

sufficient_provenance_strategy_pool = ['LLM_vanilla'] 
minimal_provenance_strategy_pool = ['null'] 

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
    #get minimal strategies with null answers and long answers to rerun 
    result = read_json(result_path)
    status = 'N/A' 
    if 'status' in result:
        status = result['status'] 
    return status 

def write_doc_sentence_size(data, data_path, embedding_folder, result_folder_path, model_name, num_case):
    objects = read_json(data_path)
    i = 0
    out_folder = '/Users/yiminglin/Documents/Codebase/doc_provenance_results/' + model_name  + '/eval' + '/' + data + '/'
    create_folder_if_not_exists(out_folder)

    size_dict = {}

    for o in objects:
        if data == 'hotpotQA':
            text = o['context']
        else: 
            text = o['text']
        i += 1

        if i > num_case:
            break

        sentences = provenance.extract_sentences_from_pdf(text)
        size_dict[i] = len(sentences) 
    write_json_to_file(out_folder + 'doc_sentence_size.json', size_dict) 

def write_doc_size(data, data_path, embedding_folder, result_folder_path, model_name, num_case):
    objects = read_json(data_path)
    i = 0
    out_folder = '/Users/yiminglin/Documents/Codebase/doc_provenance_results/' + model_name  + '/topk' + '/' + data + '/'
    create_folder_if_not_exists(out_folder)

    size_dict = {}

    for o in objects:
        if data == 'hotpotQA':
            text = o['context']
        else: 
            text = o['text']
        i += 1

        if i > num_case:
            break

        tokens = provenance.count_tokens(text)
        size_dict[i] = tokens
    write_json_to_file(out_folder + 'doc_size.json', size_dict) 


def provenance_run(data, data_path, embedding_folder, result_folder_path, model_name, num_case):
    objects = read_json(data_path)
    instruction = 'Only return answers. Do not add explanations. If answers are not found in the given context, return NULL. Context: '
    i = 0
    out_folder = '/Users/yiminglin/Documents/Codebase/doc_provenance_results/' + model_name  + '/eval' + '/' + data + '/results/'
    create_folder_if_not_exists(out_folder)

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

        if i == 349:
            continue

        if i > num_case:
            break

        for sufficient_provenance_strategy in sufficient_provenance_strategy_pool:
            for minimal_provenance_strategy in minimal_provenance_strategy_pool:
                strategy = sufficient_provenance_strategy + '_' + minimal_provenance_strategy
                
                embedding_path = get_embedding_path(data, embedding_folder, i, o)
                result_path = get_result_path(data, result_folder_path, i, o, strategy, model_name)
                out_path = get_result_path(data, out_folder, i, o, strategy, model_name)
                sufficient_path = get_sufficient_path(data, result_folder_path, i, o, sufficient_provenance_strategy, model_name)

                sufficient_time, sufficient_tokens, sufficient_provenance_ids, sufficient_eval_latency, sufficient_answers, sufficient_status =  get_sufficient_result(sufficient_path)

                out_path = out_path.replace('.json','_id.json')

                if os.path.isfile(out_path):
                    continue

                print(i, strategy)
                
                provenance.logger(text, question, title, model_name, out_path, sufficient_provenance_strategy, minimal_provenance_strategy, metric = 'LLM', embedding_path=embedding_path, sufficient_time = sufficient_time, sufficient_tokens = sufficient_tokens, sufficient_provenance_ids = sufficient_provenance_ids, sufficient_eval_latency = sufficient_eval_latency)

def provenance_run_topk(data, data_path, embedding_folder, result_folder_path, model_name, num_case):
    objects = read_json(data_path)
    i = 0
    out_folder = '/Users/yiminglin/Documents/Codebase/doc_provenance_results/' + model_name  + '/topk' + '/' + data + '/results/'
    create_folder_if_not_exists(out_folder)
    strategy = 'topk'

    for o in objects:
        if data == 'hotpotQA':
            text = o['context']
        else: 
            text = o['text']
        q = o['question']
        question = q
        i += 1
        if i > num_case:
            break

        result_path = get_result_path(data, result_folder_path, i, o, strategy, model_name)
        answer_path = result_path.replace('.json', '_answer.json')  
        if os.path.isfile(answer_path):
            continue
        
        base_strategies.divide_and_conquer_progressive_API(question, text, result_path, k=20, stop_sentence_length=5, metric = 'LLM')



def provenance_expand(data, data_path, embedding_folder, result_folder_path, model_name, num_case):
    objects = read_json(data_path)
    path = '/Users/yiminglin/Documents/Codebase/doc-provenance/tests/' + data+ '_' +  model_name + '_rerun.json'
    rerun = read_json(path)
    instruction = 'Only return answers. Do not add explanations. If answers are not found in the given context, return NULL. Context: '
    i = 0
    out_folder = '/Users/yiminglin/Documents/Codebase/doc_provenance_results/' + model_name  + '/eval' + '/' + data + '/results/'
    create_folder_if_not_exists(out_folder)

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

        if str(i) not in rerun:
            continue

        if i > num_case:
            break

        for sufficient_provenance_strategy in sufficient_provenance_strategy_pool:
            for minimal_provenance_strategy in minimal_provenance_strategy_pool:
                strategy = sufficient_provenance_strategy + '_' + minimal_provenance_strategy

                if strategy not in rerun[str(i)]:
                    continue
                
                embedding_path = get_embedding_path(data, embedding_folder, i, o)
                result_path = get_result_path(data, result_folder_path, i, o, strategy, model_name)
                out_path = get_result_path(data, out_folder, i, o, strategy, model_name)
                sufficient_path = get_sufficient_path(data, result_folder_path, i, o, sufficient_provenance_strategy, model_name)

                sufficient_time, sufficient_tokens, sufficient_provenance_ids, sufficient_eval_latency, sufficient_answers, sufficient_status =  get_sufficient_result(sufficient_path)

                if os.path.isfile(out_path):
                    continue

                print(i, strategy)
                
                provenance.logger(text, question, title, model_name, out_path, sufficient_provenance_strategy, minimal_provenance_strategy, metric = 'LLM', embedding_path=embedding_path, sufficient_time = sufficient_time, sufficient_tokens = sufficient_tokens, sufficient_provenance_ids = sufficient_provenance_ids, sufficient_eval_latency = sufficient_eval_latency)


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

def get_table_data(table_name, db_name):
    path = '/Users/yiminglin/Documents/Codebase/doc-provenance/data/64k_' + db_name + '_' + table_name + '.json'
    return read_json(path)

def get_table_text(table_names, sql_q, db_name):
    target_table = ''
    for table in table_names:
        if table in sql_q:
            target_table = table
            break 
    if target_table == '':
        print('table not found')
        return 
    table_data = get_table_data(target_table, db_name)
    concatenated_table_data = flatten_table(table_data)
    return concatenated_table_data

def provenance_tableqa_restaurant():
    model_name = 'gpt_4o_azure'
    data = 'restaurant'
    db_name = 'restaurant'
    data_file = '/Users/yiminglin/Documents/Codebase/doc-provenance/data/sampled_restaurant_queries.json'
    result_folder = '/Users/yiminglin/Documents/Codebase/doc_provenance_results/' + model_name  + '/tableqa' + '/' + data + '/results/'
    create_folder_if_not_exists(result_folder)
    embedding_path = '/Users/yiminglin/Documents/Codebase/doc-provenance/out/restaurant/embeddings/res.npy'


    data = read_json(data_file)
    provenance.set_model(model_name)
    updated_queries = []
    instruction = 'This is a table-based question. Return only the answer found in the table cell(s); do not include any explanation. The context provides each tuple as a list of column–value pairs. If answer is not found, return NULL.'

    i = 0
    for obj in data:
        type = obj['template']

        queries = obj['content']

        new_obj = {}
        new_obj['template'] = type 
        new_queries = []
        cnt = 0
        
        for query in queries:
            i += 1
            #print('query id:', i)
            nl = query['nl']
            sql = query['query']
            tables = extract_table_names(sql)
            #print(tables)
            text = get_table_text(tables, sql, db_name)
            result_path = result_folder + type + '_' + str(i) + '.json' 
            #print('result_path:', result_path)
            provenanze_sentence_size = 1
            if os.path.exists(result_path):
                result = read_json(result_path)
                if 'provenance_ids' in result:
                    provenanze_sentence_size = len(result['provenance_ids'])
            if provenanze_sentence_size == 1:
                continue 
            print('provenanze_sentence_size', i, provenanze_sentence_size)
            #print(nl, instruction)
            #print(text)
            provenance.logger(text, (nl, instruction), '', model_name, result_path, 'embedding_sufficient_top_down', 'exponential_greedy', metric = 'LLM', embedding_path=embedding_path, data_type = 'table')

if __name__ == "__main__":

    model_name = 'gpt_4o_azure'# gemini2flash
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

        print(data)
        # if data != 'paper':
        #     continue
        
        embedding_folder = parent_directory + '/out/' + data 
        result_folder = '/Users/yiminglin/Documents/Codebase/doc_provenance_results/' + model_name  + '/eval' + '/' + data + '/results/'

        print(data_folder)
        print(embedding_folder)
        print(result_folder)

        create_folder_if_not_exists(result_folder)

        provenance_run(data, data_folder, embedding_folder, result_folder, model_name, num_case)

        
    #provenance_tableqa_restaurant()