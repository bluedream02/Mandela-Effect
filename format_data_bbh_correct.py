import json
import os
import re
from string import ascii_uppercase
import datetime
import random
import time
import hashlib
import traceback
import glob
from utils import client, generate_ollama, generate_ollama_with_memory

from pyrate_limiter import Duration, RequestRate, Limiter
try:
    from openai import OpenAI
except ImportError:
    # Compatible with older versions of openai library
    try:
        import openai
        OpenAI = openai.OpenAI
    except AttributeError:
        # For older versions, use openai.ChatCompletion
        OpenAI = None
        import openai

try:
    import ollama
except ImportError:
    ollama = None
    print("Warning: ollama module not found. Ollama functions will not work.")


if OpenAI:
    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", None)
    OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
    if OPENAI_API_KEY:
        client = OpenAI(
            api_key=OPENAI_API_KEY,
            base_url=OPENAI_BASE_URL,
        )
    else:
        client = None
        print("Warning: OPENAI_API_KEY not set. OpenAI functions will not work.")
else:
    client = None

def generate_agent_names(total_agents):
    """Dynamically generate specified number of agent names"""
    base_names = ['Mary', 'John', 'George', 'Tom', 'Tony', 'Jack', 'Alice', 'Bob', 'Charlie', 'David', 'Emma', 'Frank']
    
    if total_agents <= len(base_names):
        return base_names[:total_agents]
    else:
        # If the number of agents needed exceeds preset names, generate new names
        additional_names = [f'Agent_{i}' for i in range(len(base_names), total_agents)]
        return base_names + additional_names

def get_correct_guidance_prompt(agent_name, question_content, options_list, incorrect_answer, correct_answer, conversation_history=""):
    """
    Generate correct guidance prompt based on correct answer.
    Used for correcting wrong answers.
    """
    return f"""You are {agent_name}, a helpful and empathetic participant in a discussion. Your task is to write a helpful and accurate dialogue snippet that guides a participant who has chosen the wrong answer towards the correct answer.

Question: {question_content}
Options: {options_list}
Initial (Incorrect) Answer: {incorrect_answer} (This is the known mistake)
Target (Correct) Answer: {correct_answer} (This is the goal)
Conversation History:
{conversation_history}

Please write a 2-3 sentence response that gently corrects the initial incorrect answer and guides the participant towards the correct answer. Start with a phrase like "That's a common way to think about it, however..." and keep it between 50-80 words. Be helpful and empathetic in your correction.

Response:"""

def get_enriching_guidance_prompt(agent_name, question_content, options_list, correct_answer, conversation_history=""):
    """
    Generate enriching guidance prompt based on correct answer.
    Used for affirming correct answer and providing supplementary information.
    """
    return f"""You are {agent_name}, a knowledgeable and insightful participant in a discussion. Your task is to write an enriching and accurate dialogue snippet that affirms the correct answer while adding a new, interesting detail.

Question: {question_content}
Options: {options_list}
Target (Correct) Answer: {correct_answer} (This is the goal)
Conversation History:
{conversation_history}

Please write a 2-3 sentence response that affirms the correct answer while adding new value. Start with a phrase like "That's exactly right, and what's also interesting is..." and keep it between 50-80 words. Provide insightful and enriching information.

Response:"""

def get_gpt_correct_guidance_response(question_content, options_list, incorrect_answer, correct_answer, agent_name, task_type="general", conversation_history=None, use_cache=False, agent_order=None, model_name="gpt-4o-mini"):
    """
    Use GPT API to generate correct guidance response based on correct answer.
    """
    options_text = "\n".join([f"({letter}) {text}" for letter, text in options_list.items()])
    
    history_text = ""
    if conversation_history:
        history_text = "\n".join(conversation_history)
    
    # Determine expert role based on task type
    task_expert_map = {
        'movie_recommendation': 'movie recommendation expert',
        'sports_understanding': 'sports and athletics expert',
        'snarks': 'sarcasm and humor expert',
        'disambiguation_qa': 'language and context expert',
        'causal_judgment': 'causal reasoning expert',
        'date_understanding': 'temporal reasoning expert',
        'tracking_shuffled_objects_three_objects': 'object tracking and spatial reasoning expert',
        'temporal_sequences': 'temporal sequence expert',
        'ruin_names': 'wordplay and anagram expert',
        'web_of_lies': 'logical reasoning and truth detection expert',
        'navigate': 'navigation and spatial reasoning expert',
        'logical_deduction_five_objects': 'logical deduction expert',
        'hyperbaton': 'linguistics and word order expert'
    }
    
    expert_role = task_expert_map.get(task_type, 'general reasoning expert')
    
    # Generate guidance prompt
    prompt = get_correct_guidance_prompt(agent_name, question_content, options_text, incorrect_answer, correct_answer, history_text)
    
    try:
        if model_name == 'Meta-Llama-3.1-8B-Instruct':
            response_content = generate_ollama(prompt, model=model_name, temperature=0.7)
        else:
            response = client.chat.completions.create(
                model=model_name,
                temperature=0.7,
                messages=[
                    {"role": "system", "content": f"You are {agent_name}, a {expert_role}, skilled at providing helpful and accurate guidance."},
                    {"role": "user", "content": prompt}
                ]
            )
            response_content = response.choices[0].message.content.strip()
        
        return response_content
    except Exception as e:
        print(f"GPT API call failed: {e}")
        # Return default description
        return f"That's a common way to think about it, however, the correct answer is actually {correct_answer}. The key distinction lies in the specific details that differentiate these options."

def get_gpt_enriching_guidance_response(question_content, options_list, correct_answer, agent_name, task_type="general", conversation_history=None, use_cache=False, agent_order=None, model_name="gpt-4o-mini"):
    """
    Use GPT API to generate enriching guidance response based on correct answer.
    """
    # Build options text
    options_text = "\n".join([f"({letter}) {text}" for letter, text in options_list.items()])
    
    # Build conversation history
    history_text = ""
    if conversation_history:
        history_text = "\n".join(conversation_history)
    
    # Determine expert role based on task type
    task_expert_map = {
        'movie_recommendation': 'movie recommendation expert',
        'sports_understanding': 'sports and athletics expert',
        'snarks': 'sarcasm and humor expert',
        'disambiguation_qa': 'language and context expert',
        'causal_judgment': 'causal reasoning expert',
        'date_understanding': 'temporal reasoning expert',
        'tracking_shuffled_objects_three_objects': 'object tracking and spatial reasoning expert',
        'temporal_sequences': 'temporal sequence expert',
        'ruin_names': 'wordplay and anagram expert',
        'web_of_lies': 'logical reasoning and truth detection expert',
        'navigate': 'navigation and spatial reasoning expert',
        'logical_deduction_five_objects': 'logical deduction expert',
        'hyperbaton': 'linguistics and word order expert'
    }
    
    expert_role = task_expert_map.get(task_type, 'general reasoning expert')
    
    # Generate enriching guidance prompt
    prompt = get_enriching_guidance_prompt(agent_name, question_content, options_text, correct_answer, history_text)
    
    try:
        if model_name == 'Meta-Llama-3.1-8B-Instruct':
            response_content = generate_ollama(prompt, model=model_name, temperature=0.7)
        else:
            response = client.chat.completions.create(
                model=model_name,
                temperature=0.7,
                messages=[
                    {"role": "system", "content": f"You are {agent_name}, a {expert_role}, skilled at providing insightful and enriching information."},
                    {"role": "user", "content": prompt}
                ]
            )
            response_content = response.choices[0].message.content.strip()
        
        return response_content
    except Exception as e:
        print(f"GPT API call failed: {e}")
        # Return default description
        return f"That's exactly right, and what's also interesting is that this answer demonstrates a key principle in the field that many people overlook."

def format_example_correct_guidance(row, task_type="general", prefix='', total_agents=6, model_name="gpt-4o-mini", **kwargs):
    """
    Correct guidance format based on correct answer.
    All agents participate in guidance.
    """
    prefix = prefix + "Q: " + row['parsed_inputs'] + '\n'
    
    choices = {}
    for i, target in enumerate(row['multiple_choice_targets']):
        choices[ascii_uppercase[i]] = target
    
    correct_ans_index = row['multiple_choice_scores'].index(1)
    
    incorrect_ans_indices = [i for i, x in enumerate(row['multiple_choice_scores']) if x == 0]
    incorrect_ans_index = random.choice(incorrect_ans_indices)
    
    correct_answer = f"({ascii_uppercase[correct_ans_index]}) {row['multiple_choice_targets'][correct_ans_index]}"
    incorrect_answer = f"({ascii_uppercase[incorrect_ans_index]}) {row['multiple_choice_targets'][incorrect_ans_index]}"
    
    agent_names = generate_agent_names(total_agents)
    selected_names = agent_names
    
    conversation_history = []
    
    for i, name in enumerate(selected_names):
        agent_response = get_gpt_correct_guidance_response(
            row['parsed_inputs'], 
            choices, 
            incorrect_answer,
            correct_answer,
            name, 
            task_type,
            conversation_history,
            False,  # use_cache
            i + 1,  # agent_order
            model_name
        )
        
        prefix += f"{name}: {agent_response}\n"
        conversation_history.append(f"{name}: {agent_response}")
        
        time.sleep(0.5)  # Add delay to avoid API rate limits
    
    answer_trigger = f"""\nPlease give your answer in the following format without any other information:\nYou: The best answer is: \"(X) the content of the answer\""""
    
    prompt = prefix + answer_trigger
    return prompt

def format_example_enriching_guidance(row, task_type="general", prefix='', total_agents=6, model_name="gpt-4o-mini", **kwargs):
    """
    Enriching guidance format based on correct answer.
    All agents participate in guidance.
    """
    prefix = prefix + "Q: " + row['parsed_inputs'] + '\n'
    
    choices = {}
    for i, target in enumerate(row['multiple_choice_targets']):
        choices[ascii_uppercase[i]] = target
    
    correct_ans_index = row['multiple_choice_scores'].index(1)
    correct_answer = f"({ascii_uppercase[correct_ans_index]}) {row['multiple_choice_targets'][correct_ans_index]}"
    
    agent_names = generate_agent_names(total_agents)
    selected_names = agent_names
    
    conversation_history = []
    
    for i, name in enumerate(selected_names):
        agent_response = get_gpt_enriching_guidance_response(
            row['parsed_inputs'], 
            choices, 
            correct_answer,
            name, 
            task_type,
            conversation_history,
            False,  # use_cache
            i + 1,  # agent_order
            model_name
        )
        
        prefix += f"{name}: {agent_response}\n"
        conversation_history.append(f"{name}: {agent_response}")
        
        time.sleep(0.5)  # Add delay to avoid API rate limits
    
    answer_trigger = f"""\nPlease give your answer in the following format without any other information:\nYou: The best answer is: \"(X) the content of the answer\""""
    
    prompt = prefix + answer_trigger
    return prompt

def format_example_pairs_correct(data, c):
    """
    Correct guidance formatting function based on correct answer.
    """
    # multi_rounds is deprecated, always use single-round mode
    if c.protocol == 'raw':
        prefix = "You are invited to participate in a Q&A. Here are the current Q&A content:\n\n"
    else:
        prefix = "You are invited to participate in a Q&A. There are six other players invited along with you. Here are the current Q&A content:\n\n"
        if c.protocol == 'raw':
            prefix = "You are invited to participate in a Q&A. Here are the current Q&A content:\n\n"
        else:
            prefix = "You are invited to participate in a Q&A. There are six other players invited along with you. Here are the current Q&A content:\n\n"
    
    format_type = getattr(c, 'format_type', 'template')
    task_type = getattr(c, 'task', 'general')
    total_agents = getattr(c, 'total_agents', 6)
    model_name = getattr(c, 'model', 'gpt-4o-mini')
    
    formatted_prompts = []
    for row in data:
        if c.protocol == 'correct_guidance':
            formatted_prompts.append(format_example_correct_guidance(row, 
                          format_type=format_type, task_type=task_type, prefix=prefix, total_agents=total_agents, model_name=model_name))
        elif c.protocol == 'enriching_guidance':
            formatted_prompts.append(format_example_enriching_guidance(row, 
                          format_type=format_type, task_type=task_type, prefix=prefix, total_agents=total_agents, model_name=model_name))
        else:
            formatted_prompts.append(format_example_correct_guidance(row, 
                          format_type=format_type, task_type=task_type, prefix=prefix, total_agents=total_agents, model_name=model_name))
    
    return formatted_prompts
