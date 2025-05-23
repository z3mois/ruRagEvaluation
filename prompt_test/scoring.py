from promt_test.data import (
    LLMDataset, 
    DataCollatorForLLM, 
    parse_llm_responses,
    load_full_ragbench,
    sample_dataset,
    has_sentences,
    under_token_limit
)
from promt_test.utils import (
    send_async_requests
)
from torch.utils.data import DataLoader
from promt_test.metrics import (
    evaluate_dataset
)
import pickle
import asyncio
from tqdm import tqdm
import numpy as np
MODEL_NAME = "Qwen/Qwen2.5-72B-Instruct"
DELTA = 20
async def evalute_df(model_id, client, ds, article_prompt, tokenizer):
    max_tokens=1024
    dataset_prompt = LLMDataset(ds, article_prompt, tokenizer, max_length=14000-max_tokens-DELTA)
    collator = DataCollatorForLLM()
    dataloader = DataLoader(dataset_prompt, batch_size=8, shuffle=False, collate_fn=DataCollatorForLLM() )
    res = []
    for elem in tqdm(dataloader):
        temp_responses = await send_async_requests(elem, model=model_id, client=client,max_tokens=max_tokens)
        res.extend(temp_responses)
            
        
    return res, dataset_prompt.get_stats()


async def run_evaluation(article_prompt, client, model_id, tokenizer, sample_percent=5):
    ragbench = load_full_ragbench()

    results = {}
    bad = {}
    for name, dataset in ragbench.items():
        results[name] = {}
        bad[name] = {}
        for subset in ['train', 'validation', 'test']:
            if subset in dataset:
                true_len = len(dataset[subset])
                print(f'Датасет {name}.{subset}: {len(dataset[subset])}')
                dataset[subset] = dataset[subset].filter(has_sentences)
                print(f'Датасет {name}.{subset} после чистки: {len(dataset[subset])} на пусто')
                dataset[subset] = dataset[subset].filter(lambda x: under_token_limit(x, article_prompt, tokenizer, max_tokens=8000))
                print(f'Датасет {name}.{subset} после чистки: {len(dataset[subset])} на max_tokens')
                delta = true_len - len(dataset[subset])
                print(f'Датасет {name}.{subset} после чистки: {len(dataset[subset])}')
                sampled_subset = sample_dataset(dataset[subset], sample_percent)
                print(f'Взято из {name}.{subset} после чистки: {len(sampled_subset)}')
                preds, token_len_error = await evalute_df(model_id, client, sampled_subset, article_prompt, tokenizer)
#                 print(token_len_error)
                parsed_samples, failed_samples = parse_llm_responses(preds)

                print(f'Успешно распарсено в {name}.{subset}: {len(parsed_samples)}')
                print(f'Ошибочных ответов в {name}.{subset}: {len(failed_samples)}')
                
                failed_idxs = {f['index'] for f in failed_samples}
                filtered_ground_truths = [gt for idx, gt in enumerate(sampled_subset) if idx not in failed_idxs]
                ground_truths = [{'all_utilized_sentence_keys': elem['all_utilized_sentence_keys'],
                  'all_relevant_sentence_keys': elem['all_relevant_sentence_keys']
                 } for elem in filtered_ground_truths]
                metrics = evaluate_dataset(parsed_samples, filtered_ground_truths)
                for_save = {'failed_idxs_after':failed_idxs, 'parsed_answer': parsed_samples, 'preds': preds}
                with open(f'../tmp/{name}.{subset}.pkl', 'wb') as f:
                    pickle.dump(for_save, f)
                results[name][subset] = metrics
                bad[name][subset] = {}

                bad[name][subset]['llm_error'] = len(failed_samples)
                
                bad[name][subset]['token_len_error'] = token_len_error
                bad[name][subset]['dataset_error'] = delta
                for_save_tmp = {'bad':bad, 'results': results}
                with open(f'../tmp/tmp_res.pkl', 'wb') as f:
                    pickle.dump(for_save_tmp, f)
    relevant_overlaps, utilized_overlaps = [], []
    relevant_misses, utilized_misses = [], []
    relevant_extras, utilized_extras = [], []

    for dataset_results in results.values():
        for metrics in dataset_results.values():
            relevant_overlaps.append(metrics['relevant_overlap_mean'])
            utilized_overlaps.append(metrics['utilized_overlap_mean'])
            relevant_misses.append(metrics['relevant_miss_mean'])
            utilized_misses.append(metrics['utilized_miss_mean'])
            relevant_extras.append(metrics['relevant_extra_mean'])
            utilized_extras.append(metrics['utilized_extra_mean'])

    aggregated_metrics = {
        'total_relevant_overlap_mean': np.mean(relevant_overlaps),
        'total_relevant_miss_mean': np.mean(relevant_misses),
        'total_relevant_extra_mean': np.mean(relevant_extras),
        'total_utilized_overlap_mean': np.mean(utilized_overlaps),
        'total_utilized_miss_mean': np.mean(utilized_misses),
        'total_utilized_extra_mean': np.mean(utilized_extras),
    }

    return results, aggregated_metrics, bad

