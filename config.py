# Configuration file which is used by the apps to define the parameters of the blog Summarizer

Config = {
    'summarizer_args': {'llm_args': {'llm_provider': 'llama-index-togetherai',  # llama-index-huggingface, llama-index-openai
                                     'llm_model_name': 'mistralai/Mixtral-8x7B-Instruct-v0.1',
                                     # meta-llama/Llama-2-7b-chat-hf, gpt-3.5-turbo
                                     'llm_model_path': '/Users/bhargavvankayalapati/Work/BlogSummarizer/BlogSummarizer/Models/meta-llama/Llama-2-7b-chat-hf',
                                     # required if using locally downloaded model
                                     'offload_dir': './offload_dir',
                                     'cache_dir': '/Users/bhargavvankayalapati/Work/InHouseRAG/InHouseRAG/Models/meta-llama/Llama-2-7b-chat-hf',
                                     # required if using locally downloaded model
                                     'local_files_only': True, 'context_window': 4096,
                                     'max_new_tokens': 512,
                                     'generate_kwargs': {"temperature": 0.7, "top_k": 50, "top_p": 0.95,
                                                         'do_sample': False},
                                     'tokenizer_max_length': 4096,
                                     'stopping_ids': (50278, 50279, 50277, 1, 0), },
                        'refetch_blogs': False,  # To avoid refetching the blog content from the provided blogs URL.
                        'output_dir': 'Data/Blogs_content',
                        'observ_provider': 'phoenix',
                        },
    'query_engine_args': {'query_engine_type': 'RetrieverQueryEngine',
                          'query_engine_kwargs': None,
                          'response_mode': 'tree_summarize', 'chunk_size': 512,
                          'chunk_overlap': 64,
                          'streaming': True,
                          'summary_template_str': "The contents of a blog titled {query_str} are provided as Context information below.\n"
                                                  "---------------------\n"
                                                  "{context_str}\n"
                                                  "---------------------\n"
                                                  "Given the information and not prior knowledge, summarize the blog.\n"
                                                  "Summary: ",
                          # Using a custom summary template to help generate summaries. This prompt can be optimized
                          # for an optimized response from the LLM.
                          'use_async': False},
}
