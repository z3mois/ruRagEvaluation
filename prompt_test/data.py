from datasets import load_dataset
import random
import numpy as np
from tqdm import tqdm 
DATASETS = ['covidqa']#, 'cuad', 'delucionqa', 'emanual', 'expertqa', 'finqa', 'hagrid', 'hotpotqa', 'msmarco', 'pubmedqa', 'tatqa', 'techqa']

def load_full_ragbench():
    ragbench = {}
    for dataset in DATASETS:
        ragbench[dataset] = load_dataset("rungalileo/ragbench", dataset)
    return ragbench


def sample_dataset(dataset, percent):
    sample_size = int(len(dataset) * percent / 100)
    return dataset.select(random.sample(range(len(dataset)), sample_size))

class LLMDataset:
    def __init__(self, data, article_prompt, tokenizer, max_length=14000):
        """
        :param data: Список элементов датасета.
        :param article_prompt: Строка с форматированием для создания финального prompt.
        """
        self.data = data
        self.article_prompt = article_prompt
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.stats = 0
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        documents = ''
        for doc in item['documents_sentences']:
            for sentence in doc:
                documents += ' ' + ' '.join(sentence)
            documents += '\n'  
        filled_prompt = self.article_prompt.format(
            documents=documents,
            question=item['question'],
            answer=item['response']
        )
        

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": filled_prompt}
        ]
#         system_content = messages[0]['content']
#         user_content = messages[1]['content']
#         system_tokens = self.tokenizer.encode(system_content, add_special_tokens=False)
#         user_tokens = self.tokenizer.encode(user_content, add_special_tokens=False)
#         total_tokens = len(system_tokens) + len(user_tokens)
        
#         if total_tokens > self.max_length:
#             self.stats += 1
#             max_user_tokens = self.max_length - len(system_tokens)
#             if max_user_tokens > 0:
#                 # с конца 
#                 truncated_tokens = user_tokens[-max_user_tokens:]
#                 messages[1]['content'] = self.tokenizer.decode(truncated_tokens)
#             else:
#                 # В крайнем случае обрезаем системное сообщение
#                 truncated_system = self.tokenizer.decode(system_tokens[:self.max_length])
#                 messages = [{"role": "system", "content": truncated_system}]
        return messages
    def get_stats(self):
        return self.stats
class DataCollatorForLLM:
    def __call__(self, batch):
        """
        Принимает батч, где каждый элемент — это список месседжей 
        Возвращает единый список месседжей из всех элементов батча.
        """
        flattened_messages = []
        for sample in batch:
            flattened_messages.append(sample)
        return flattened_messages

def parse_llm_responses(responses):
    successful_samples = []
    failed_samples = []

    for idx, elem in enumerate(tqdm(responses)):
        original_elem = elem
        elem = elem.replace(' true', 'True')
        elem = elem.replace(' false', 'False')
        elem = elem.replace('```json', '')
        elem = elem.replace('```', '')
        elem = elem.replace('‘‘‘', '')
        try:
            parsed_elem = eval(elem)
            successful_samples.append(parsed_elem)
        except Exception as e:
            elem_fixed = elem + '}'
            try:
                parsed_elem = eval(elem_fixed)
                successful_samples.append(parsed_elem)
            except Exception as ee:
                failed_samples.append({
                    'original_response': original_elem,
                    'fixed_response_attempt': elem_fixed,
                    'error': str(ee),
                    'index': idx
                })

    return successful_samples, failed_samples

def has_sentences(example):
    """
    Функция возвращает True, если пример содержит хотя бы одно предложение
    в documents_sentences (и, опционально, в response_sentences).
    Иначе возвращает False.
    """

    if "documents_sentences" not in example:
        return False
    if not isinstance(example["documents_sentences"], list):
        return False
    total_doc_sentences = 0
    for doc in example["documents_sentences"]:
        if isinstance(doc, list):
            total_doc_sentences += len(doc)
    if total_doc_sentences == 0:
        return False

    if "response_sentences" in example:
        if not isinstance(example["response_sentences"], list):
            return False
        if len(example["response_sentences"]) == 0:
            return False
    if 'question' and 'response' not in example:
        False
    return True

def make_messages(item, article_prompt):
    # собираем message
    documents = ""
    for doc in item['documents_sentences']:
        for sentence in doc:
            documents += " " + " ".join(sentence)
        documents += "\n"
    filled = article_prompt.format(
        documents=documents,
        question=item['question'],
        answer=item['response']
    )
    return [
        {"role":"system","content":"You are a helpful assistant."},
        {"role":"user","content":filled}
    ]

def under_token_limit(item, prompt, tokenizer, max_tokens=8000):
    messages = make_messages(item, prompt)
    encoded = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        truncation=False,
        return_tensors="pt"
    )
    # shape == (1, seq_len)
    seq_len = encoded.shape[1]
    return seq_len <= max_tokens