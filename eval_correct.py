from time import time
from string import ascii_uppercase
import traceback
import re
import json
import glob
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from collections import defaultdict
try:
    import ollama  # Enable ollama support for local models
except ImportError:
    ollama = None
    print("Warning: ollama module not found. Ollama functions will not work.")
import openai
import argparse
import pandas as pd
from openpyxl import load_workbook
from openpyxl.styles import Font, Color
from utils import Config, SEP, generate_gpt, generate_gpt_empowered, generate_ollama, generate_ollama_empowered, generate_gpt_with_memory, generate_gpt_empowered_with_memory, generate_ollama_with_memory, generate_ollama_empowered_with_memory, generate_gpt_with_memory_followup
from format_data_bbh_combined import format_example_pairs
from format_data_bbh_correct import format_example_pairs_correct


parser = argparse.ArgumentParser()
parser.add_argument('--total_agents', default=5, type=int, help='total number of agents (>=1 and <=15, recommended >=3)')
parser.add_argument('--model', default='gpt-4o-mini', type=str) # , required=True)
parser.add_argument('--save_path', default='output_correct', type=str) #, required=True)

parser.add_argument('--dataset', default='bbh', type=str)
parser.add_argument('--use_cache', default=True, type=lambda x: x.lower() == 'true' if isinstance(x, str) else bool(x), help='Use cache for agent responses (default: True). Set to False to disable caching.')
parser.add_argument('--memory_mode', default=True, action='store_true', help='Enable memory mode: agent first answers, then summarizes memory')
parser.add_argument('--max_workers', default=100, type=int, help='Maximum number of worker threads for processing multiple tasks')
parser.add_argument('--data_folder', default='bbh_all_small', type=str, help='Data folder to process: bbh_select_small, bbh_all_small, or custom folder name')

args = parser.parse_args()

ans_map = {k: v for k,v in zip(ascii_uppercase, range(26))}

# Set to true to run on a small subset of the data
testing = True

# Data folder configuration - can be modified here directly or specified via command line argument --data_folder
DATA_FOLDER = args.data_folder  # Default to command line argument, can also be hardcoded here

def get_all_tasks_from_data_folder():
    """Get all tasks from the specified data folder"""
    data_folder_path = f"data/{DATA_FOLDER}"
    if not os.path.exists(data_folder_path):
        print(f"❌ Path {data_folder_path} does not exist")
        return []
    
    task_dirs = [d for d in os.listdir(data_folder_path) 
                 if os.path.isdir(os.path.join(data_folder_path, d))]
    
    # Sort alphabetically
    task_dirs.sort()
    
    print(f"🔍 Found {len(task_dirs)} tasks from {DATA_FOLDER}:")
    for i, task in enumerate(task_dirs, 1):
        print(f"   {i:2d}. {task}")
    
    return task_dirs

def check_if_output_exists(task, config_template, args):
    """Check if output files already exist"""
    try:
        config = Config(task, 
                    protocol=config_template['protocol'],
                    # majority_num已移除，在Config类中自动设置为total_agents
                    total_agents=config_template['total_agents'],
                    model=config_template['model'],
                    batch=config_template['batch'],
                    use_cache=config_template['use_cache'],
                    memory_mode=config_template['memory_mode'])
        
        config.fname = str(config) + '.json'
        save_path = args.save_path
        xlsx_path = f'{save_path}/{config.fname[:-5]}.xlsx'
        json_path = f'{save_path}/{config.fname}'
        
        xlsx_exists = os.path.exists(xlsx_path)
        json_exists = os.path.exists(json_path)
        
        return {
            'xlsx_exists': xlsx_exists,
            'json_exists': json_exists,
            'xlsx_path': xlsx_path,
            'json_path': json_path,
            'config': config
        }
        
    except Exception as e:
        print(f"❌ Error checking output files for task {task}: {str(e)}")
        return None

def process_single_task(task, config_template, args):
    """Function to process a single task"""
    try:
        print(f"\n🚀 Starting to process task: {task}")
        
        # First check if output files already exist
        check_result = check_if_output_exists(task, config_template, args)
        if check_result is None:
            print(f"❌ Unable to check output file status for task {task}")
            return None
        
        config = check_result['config']
        xlsx_exists = check_result['xlsx_exists']
        json_exists = check_result['json_exists']
        xlsx_path = check_result['xlsx_path']
        json_path = check_result['json_path']
        
        # If xlsx file already exists, skip processing
        if xlsx_exists:
            print(f"⏭️  XLSX file for task {task} already exists, skipping")
            print(f"   📁 XLSX path: {xlsx_path}")
            if json_exists:
                print(f"   📁 JSON path: {json_path}")
            return {
                'task': task,
                'config': config.__dict__,
                'skipped': True,
                'reason': 'xlsx_file_exists'
            }
        
        print(f"📋 Configuration: {config.__dict__}")
        print(f"📁 Will generate file: {xlsx_path}")
        
        # Load data
        if args.dataset == 'bbh':
            # Use bbh_all_small dataset
            data_path = f'data/{DATA_FOLDER}/{task}/val_data.json'
            
            if not os.path.exists(data_path):
                print(f"❌ Data file does not exist: {data_path}")
                return None
            
            with open(data_path, 'r') as f:
                data = json.load(f)['data']
        
        # Test mode: only process first 5 samples
        if testing:
            print('🧪 Test mode')
            data = data[:1]
            print(f'Test mode: processing only {len(data)} samples')
            if args.use_cache:
                print('Note: Cache in test mode may be incomplete, recommend regenerating cache in full mode')
        
        format_inps = format_example_pairs(data, config)
        outputs = defaultdict(lambda: [None for _ in range(len(data))])
        idx_list = range(len(data))
        failed_idx = []
        
        print(f"📊 Starting to process {len(data)} instances...")
        
        future_instance_outputs = {}
        batch = 1 if not hasattr(config, 'batch') else config.batch
        
        with ThreadPoolExecutor(max_workers=batch) as executor:
            for idx in idx_list:
                future_instance_outputs[executor.submit(get_results_on_instance_i, idx, format_inps, data, config, failed_idx)] = idx 
            
            for cnt, instance_outputs in enumerate(tqdm(as_completed(future_instance_outputs), total=len(future_instance_outputs), desc=f"Processing {task}")):
                i = future_instance_outputs[instance_outputs]
                kv_outputs_list = instance_outputs.result(timeout=500)
                kv_outputs = kv_outputs_list[0]
                for key, val in kv_outputs.items():
                    outputs[key][i] = val
                
                # Save results every 100 instances or when completed
                if cnt + 1 % 100 == 0 or cnt + 1 == len(idx_list):
                    print(f'=== {task} Progress: ', cnt + 1, '/', len(idx_list), '===')
                    
                    acc = sum([int(y==z) for y,z in zip(outputs['y_pred'], outputs['y_true']) if y is not None and z is not None])
                    print(f'{task} Accuracy:', acc)
                    print(f'{task} Failed count:', len(failed_idx))
                    
                    # Save results
                    save_results(config, outputs, failed_idx, args, task)
        
        print(f"✅ Task {task} processing completed")
        return {
            'task': task,
            'config': config.__dict__,
            'outputs': outputs,
            'failed_idx': failed_idx,
            'accuracy': acc if 'acc' in locals() else 0
        }
        
    except Exception as e:
        print(f"❌ Error processing task {task}: {str(e)}")
        traceback.print_exc()
        return None

def save_results(config, outputs, failed_idx, args, task):
    """Save results to file"""
    try:
        save_path = args.save_path
        os.makedirs(save_path, exist_ok=True)
        
        fname = config.fname
        with open(f'{save_path}/{fname}','w') as f:
            json.dump({
                'config': config.__dict__,
                'fname': fname,
                'failed_idx': failed_idx,
                'outputs': outputs,
            }, f)
        
        xlsx_path = f'{save_path}/{fname[:-5]}.xlsx'
        save_to_xlsx({
            'config': config.__dict__,
            'failed_idx': failed_idx,
            'outputs': outputs,
        }, xlsx_path)
        
        print(f"💾 Results for {task} saved to: {save_path}")
        
    except Exception as e:
        print(f"❌ Error saving results for {task}: {str(e)}")

def extract_answer(model_answer):
    try:
        
        tmp=model_answer.split('is: "(')
        if len(tmp) == 1:
            tmp = model_answer.split('is: (')
        if len(tmp) == 1:
            tmp = model_answer.split('is (')
        assert len(tmp) > 1, "model didn't output trigger"
        assert tmp[-1][1] == ')', "didnt output letter for choice"
        pred = tmp[-1][0]
        return pred
    except Exception as e:
        return traceback.format_exc()

def extract_answer_and_memory(model_answer):
    """
    Extract both answer and memory from model response.
    Returns: (pred, memory)
    """
    try:
        # First try standard format to extract answer
        try:
            pred = extract_answer(model_answer)
        except:
            pred = None
        
        # If standard format fails, try Chinese format
        if pred is None or "Traceback" in str(pred):
            if '答案:' in model_answer:
                answer_part = model_answer.split('答案:')[1]
                if 'Memory:' in answer_part:
                    answer_part = answer_part.split('Memory:')[0]
                
                # Try to extract letter in parentheses
                import re
                match = re.search(r'\(([A-Z])\)', answer_part)
                if match:
                    pred = match.group(1)
                else:
                    pred = "X"  # Default value
            else:
                pred = "X"
        
        memory = None
        if 'Memory:' in model_answer:
            memory_parts = model_answer.split('Memory:')
            if len(memory_parts) > 1:
                memory = memory_parts[1].strip()
                memory = memory.replace('\n', ' ').strip()
        
        return pred, memory
    except Exception as e:
        return "X", None

def save_to_xlsx(data, xlsx_path):
    
    flattened_data = {}
    for key, value in data.items():
        if isinstance(value, dict):
            for nested_key, nested_value in value.items():
                flattened_data[nested_key] = nested_value
        else:
            flattened_data[key] = value
    
    df = pd.DataFrame([flattened_data])
    
    delete_key_lst = ['batch', 'fname']
    for delete_key in delete_key_lst:
        if delete_key in df.columns:
            df = df.drop([delete_key], axis=1)
    
    
    # Process two groups of experimental data
    if 'raw_inputs' in df.columns and isinstance(df['raw_inputs'].values[0], list):
        df = df.reindex(df.index.repeat(len(df['raw_inputs'].values[0])))
        
        # Define column names for two groups of experiments
        experiment_columns = [
            # Experiment 1: Baseline Reality Protocol
            ['raw_inputs', 'raw_outputs', 'raw_y_pred', 'raw_y_true', 'raw_is_correct'],
            # Experiment 2: Correct Guidance Protocol
            ['correct_guidance_inputs', 'correct_guidance_outputs', 'correct_guidance_y_pred', 'correct_guidance_y_true', 'correct_guidance_is_correct'],
        ]
        
        # If memory mode is enabled, add memory-related columns
        memory_mode = False
        if 'config' in data and 'memory_mode' in data['config']:
            memory_mode = data['config']['memory_mode']
        
        if memory_mode:
            experiment_columns[1].append('correct_guidance_memory')  # Experiment 2 adds memory
        
        for experiment_cols in experiment_columns:
            for col in experiment_cols:
                if col in df.columns:
                    first_value = df[col].values[0]
                    if isinstance(first_value, list):
                        for i in range(len(first_value)):
                            df.iloc[i, df.columns.get_loc(col)] = first_value[i]
    
    # Reorder columns to make two groups of experimental data more logical
    column_order = []
    
    # Experiment 1: Baseline Reality Protocol
    if 'raw_inputs' in df.columns:
        column_order.extend(['raw_inputs', 'raw_outputs', 'raw_y_pred', 'raw_y_true', 'raw_is_correct'])
    
    # Experiment 2: Correct Guidance Protocol
    if 'correct_guidance_inputs' in df.columns:
        column_order.extend(['correct_guidance_inputs', 'correct_guidance_outputs', 'correct_guidance_y_pred', 'correct_guidance_y_true', 'correct_guidance_is_correct'])
        if 'correct_guidance_memory' in df.columns:
            column_order.append('correct_guidance_memory')
    
    config_columns = [col for col in df.columns if col not in column_order]
    column_order.extend(config_columns)
    
    df = df[column_order]
    
    df.to_excel(xlsx_path, index=False)
    
    # Try to adjust column width, if failed only save basic Excel file
    try:
        wb = load_workbook(xlsx_path)
        ws = wb.active
        
        for column_cells in ws.columns:
            key_length = len(str(column_cells[0].value))
            value_length = max([len(str(cell.value)) for cell in column_cells[1:] if cell.value is not None], default=0)
            column_letter = column_cells[0].column
            ws.column_dimensions[column_letter].width = min(max(key_length*2, value_length+5), 60)
        
        wb.save(xlsx_path)
        print(f"Excel file saved successfully (with column width adjustment): {xlsx_path}")
    except Exception as e:
        print(f"Column width adjustment failed, but Excel file saved: {xlsx_path}")
        print(f"Error message: {e}")
    
    return df

def get_results_on_instance_i(i, format_inps, data, c, failed_idx):
    """
    Two groups of experiments comparison:
    1. Baseline Reality Protocol: Direct question asking
    2. Correct Guidance Protocol: Determine guidance method based on correctness of protocol 1's answer
    """
    kv_outputs_list = []

    inp = format_inps[i]
    y_true = data[i]['multiple_choice_scores'].index(1)
    row = data[i]

    choices = {}
    for j, choice in enumerate(row['multiple_choice_targets']):
        choices[ascii_uppercase[j]] = choice

    # Experiment 1: Baseline Reality Protocol - Direct question asking (without agent discussion)
    if "Answer choices:" in row['parsed_inputs']:
        # If options are already included, use directly
        raw_question = f"""Q: {row['parsed_inputs']}

Please give your answer in the following format without any other information:
You: The best answer is: "(X) the content of the answer\""""
    else:
        # If options are not included, add them
        raw_question = f"""Q: {row['parsed_inputs']}

Answer choices:
{chr(10).join([f"({letter}) {text}" for letter, text in choices.items()])}

Please give your answer in the following format without any other information:
You: The best answer is: "(X) the content of the answer\""""
    
    try:
        if c.model == 'Meta-Llama-3.1-8B-Instruct':
            raw_output = generate_ollama(raw_question, model=c.model, temperature=.7)
        else:
            raw_output = generate_gpt(raw_question, model=c.model, temperature=.7)
        raw_pred = extract_answer(raw_output)
        raw_y_pred = int(ans_map.get(raw_pred, -1))
    except Exception as e:
        print(f"Baseline Reality Protocol experiment failed for instance {i}: {e}")
        raw_output = "Experiment failed"
        raw_y_pred = -1

    # Experiment 2: Correct Guidance Protocol
    try:
        if raw_y_pred == y_true:
            print(f"Experiment 2 - Protocol 1 answer is correct, using rich guidance")
            correct_guidance_config = Config(c.task, 
                                          protocol='enriching_guidance',
                                          # majority_num has been removed, automatically set to total_agents in Config class
                                          total_agents=c.total_agents,
                                          model=c.model,
                                          batch=c.batch,
                                          use_cache=c.use_cache,
                                          memory_mode=c.memory_mode)
        else:
            print(f"Experiment 2 - Protocol 1 answer is wrong, using corrective guidance")
            correct_guidance_config = Config(c.task, 
                                          protocol='correct_guidance',
                                          # majority_num has been removed, automatically set to total_agents in Config class
                                          total_agents=c.total_agents,
                                          model=c.model,
                                          batch=c.batch,
                                          use_cache=c.use_cache,
                                          memory_mode=c.memory_mode)
        
        # Use new formatting function to generate correct guidance prompt
        correct_guidance_format_inps = format_example_pairs_correct([row], correct_guidance_config)
        correct_guidance_inp =         correct_guidance_format_inps[0]
        
        if args.memory_mode:
            if c.model == 'Meta-Llama-3.1-8B-Instruct':
                correct_guidance_out = generate_ollama_with_memory(correct_guidance_inp, model=c.model, temperature=.7)
            else:
                correct_guidance_out = generate_gpt_with_memory(correct_guidance_inp, model=c.model, temperature=.7)
            
            correct_guidance_pred, correct_guidance_memory = extract_answer_and_memory(correct_guidance_out)
            correct_guidance_y_pred = int(ans_map.get(correct_guidance_pred, -1))
            
            print(f"Experiment 2 - Model final answer (with memory): {correct_guidance_out}")
            print(f"Experiment 2 - Extracted memory: {correct_guidance_memory}")
        else:
            if c.model == 'Meta-Llama-3.1-8B-Instruct':
                correct_guidance_out = generate_ollama(correct_guidance_inp, model=c.model, temperature=.7)
            else:
                correct_guidance_out = generate_gpt(correct_guidance_inp, model=c.model, temperature=.7)
            
            correct_guidance_pred = extract_answer(correct_guidance_out)
            correct_guidance_y_pred = int(ans_map.get(correct_guidance_pred, -1))
            correct_guidance_memory = None
            
            print(f"Experiment 2 - Model final answer: {correct_guidance_out}")
        
    except Exception as e:
        print(f"Correct Guidance Protocol experiment failed for instance {i}: {e}")
        correct_guidance_inp = "Experiment failed"
        correct_guidance_out = "Experiment failed"
        correct_guidance_y_pred = -1
        correct_guidance_memory = "Experiment failed"

    kv_outputs = {
        # Experiment 1: Baseline Reality Protocol
        'raw_inputs': raw_question,
        'raw_outputs': raw_output,
        'raw_y_pred': raw_y_pred,
        'raw_y_true': y_true,
        'raw_is_correct': raw_y_pred == y_true,
        
        # Experiment 2: Correct Guidance Protocol
        'correct_guidance_inputs': correct_guidance_inp,  # Record input prompt sent to GPT
        'correct_guidance_outputs': correct_guidance_out,  # Record actual GPT response
        'correct_guidance_y_pred': correct_guidance_y_pred,
        'correct_guidance_y_true': y_true,
        'correct_guidance_is_correct': correct_guidance_y_pred == y_true,
    }
    
    # If memory mode is enabled, add memory field
    if args.memory_mode and correct_guidance_memory:
        kv_outputs['correct_guidance_memory'] = correct_guidance_memory
    
    kv_outputs_list.append(kv_outputs)

    return kv_outputs_list

def main():
    # use this to retry examples that previously failed
    # List paths to the json files for the results you want to retry
    configs_to_resolve = []  # Add this variable definition
    
    # Validate total_agents parameter
    if args.total_agents < 1:
        raise ValueError("total_agents must be >= 1")
    if args.total_agents > 15:
        raise ValueError("total_agents > 15")
    # All agents choose wrong answer, so majority_num equals total_agents
    # No longer need separate majority_num parameter validation
    
    # Display currently used data folder
    print(f"📁 Currently using data folder: {DATA_FOLDER}")
    
    # Only process specified task folders
    # 'english_proverbs',
    # 'physics_questions',
    # 'temporal_sequences',

    target_tasks = [
        'anachronisms',
        'causal_judgment', 
        'disambiguation_qa',
        # 'dyck_languages',
        # 'empirical_judgments',
        # 'epistemic_reasoning',
        # 'general_knowledge',
        # 'international_phonetic_alphabet_nli',
        # 'known_unknowns',
        # 'language_identification',
        # 'misconceptions',
        # 'movie_recommendation',
        # 'presuppositions_as_nli',
        # 'qa_wikidata',
        # 'salient_translation_error_detection',
        # 'sports_understanding',
        # 'tellmewhy',
        # 'vitaminc_fact_verification',
        # 'which_wiki_edit',
        # 'auto_categorization'
    ]


    
    print(f"\n🎯 Specified mode: processing the following {len(target_tasks)} tasks:")
    for i, task in enumerate(target_tasks, 1):
        print(f"   {i:2d}. {task}")
    
    config_template = {
        'protocol': 'raw',  # Baseline Reality Protocol
        # majority_num has been removed, automatically set to total_agents in Config class
        'total_agents': args.total_agents,
        'model': args.model,
        'batch': 5,
        'use_cache': args.use_cache,
        'memory_mode': args.memory_mode
    }
    
    print(f"\n🔧 Configuration template:")
    for key, value in config_template.items():
        print(f"   {key}: {value}")
    
    print(f"\n🚀 Starting to process specified tasks...")
    print(f"   Maximum worker threads: {args.max_workers}")
    
    print(f"\n🔍 Pre-check phase: Checking output file status...")
    tasks_to_process = []
    tasks_to_skip = []
    
    for task in target_tasks:
        check_result = check_if_output_exists(task, config_template, args)
        if check_result is None:
            print(f"❌ Unable to check task {task}, will attempt to process")
            tasks_to_process.append(task)
        elif check_result['xlsx_exists']:
            tasks_to_skip.append(task)
            print(f"⏭️  {task}: XLSX file already exists, skipping")
        else:
            tasks_to_process.append(task)
            print(f"🔄 {task}: Needs processing")
    
    print(f"\n📊 Pre-check results:")
    print(f"   Tasks to process: {len(tasks_to_process)}")
    print(f"   Tasks to skip: {len(tasks_to_skip)}")
    
    if tasks_to_skip:
        print(f"\n⏭️  Skipped tasks:")
        for task in tasks_to_skip:
            print(f"   • {task}")
    
    if not tasks_to_process:
        print(f"\n🎉 All tasks have been processed, no need to rerun!")
        return
    
    print(f"\n🔄 Tasks to process:")
    for task in tasks_to_process:
        print(f"   • {task}")
    
    completed_tasks = []
    failed_tasks = []
    
    print(f"\n🚀 Starting task-level concurrent processing, maximum concurrent tasks: {args.max_workers}")
    
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        future_tasks = {}
        for task in tasks_to_process:
            future = executor.submit(process_single_task, task, config_template, args)
            future_tasks[future] = task
            print(f"📋 Submitted task: {task}")
        
        print(f"\n🔄 All {len(tasks_to_process)} tasks to process have been submitted to thread pool, starting concurrent execution...")
        
        for i, future in enumerate(as_completed(future_tasks), 1):
            task = future_tasks[future]
            print(f"\n📋 [{i:2d}/{len(tasks_to_process)}] Task completed: {task}")
            
            try:
                result = future.result(timeout=3600)  # 1 hour timeout
                if result:
                    if result.get('skipped', False):
                        print(f"⏭️  {task} skipped")
                        tasks_to_skip.append(task)
                    else:
                        print(f"✅ {task} completed")
                        completed_tasks.append(task)
                else:
                    print(f"❌ {task} failed")
                    failed_tasks.append(task)
            except Exception as e:
                print(f"❌ {task} execution exception: {str(e)}")
                failed_tasks.append(task)
    
    # Display processing result statistics
    print(f"\n📊 Task processing result statistics:")
    print(f"   Total tasks: {len(target_tasks)}")
    print(f"   Skipped tasks: {len(tasks_to_skip)}")
    print(f"   Successfully completed: {len(completed_tasks)}")
    print(f"   Failed tasks: {len(failed_tasks)}")
    
    if tasks_to_skip:
        print(f"\n⏭️  Skipped tasks (files already exist):")
        for task in tasks_to_skip:
            print(f"   • {task}")
    
    if completed_tasks:
        print(f"\n✅ Successfully completed tasks:")
        for task in completed_tasks:
            print(f"   • {task}")
    
    if failed_tasks:
        print(f"\n❌ Failed tasks:")
        for task in failed_tasks:
            print(f"   • {task}")
    
    print(f"\n🎉 All specified tasks processing completed!")
    
    print(f'\n⏱️  Total time: {round(time() - first_start)} seconds')

if __name__ == '__main__':
    first_start = time()  # Record start time
    main()