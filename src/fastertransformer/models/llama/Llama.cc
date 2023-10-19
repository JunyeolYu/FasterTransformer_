/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "src/fastertransformer/models/llama/Llama.h"
#include "src/fastertransformer/kernels/bert_preprocess_kernels.h"
#include "src/fastertransformer/kernels/decoding_kernels.h"
#include "src/fastertransformer/kernels/gpt_kernels.h"
#include "src/fastertransformer/layers/beam_search_layers/BaseBeamSearchLayer.h"
#include <algorithm>

namespace fastertransformer {

template<typename T>
void Llama<T>::initialize()
{
    gpt_context_decoder_ = new LlamaContextDecoder<T>(head_num_,
                                                      size_per_head_,
                                                      inter_size_,
                                                      num_layer_,
                                                      rotary_embedding_dim_,
                                                      neox_rotary_style_,
                                                      use_gptj_residual_,
                                                      layernorm_eps_,
                                                      tensor_para_,
                                                      pipeline_para_,
                                                      stream_,
                                                      cublas_wrapper_,
                                                      allocator_,
                                                      is_free_buffer_after_forward_,
                                                      is_context_qk_buf_float_,
                                                      attention_type_,
                                                      custom_all_reduce_comm_,
                                                      enable_custom_all_reduce_);

    // parse env overrides
    if (std::getenv("LLAMA_STREAM_CB_STEP") != nullptr) {
        try {
            int callback_step_from_env = stoi(
                std::string(std::getenv("LLAMA_STREAM_CB_STEP"))
                );
            token_generated_cb_step_ = callback_step_from_env;
            FT_LOG_INFO("Override stream callback step to %d from LLAMA_STREAM_CB_STEP",
                token_generated_cb_step_);
        } catch (...) {
            FT_LOG_WARNING("convert LLAMA_STREAM_CB_STEP err, use default value %d",
                token_generated_cb_step_);
        }
    }
}

template<typename T>
void Llama<T>::allocateBuffer()
{
    FT_CHECK(false);
}

template<typename T>
void Llama<T>::allocateBuffer(
    size_t batch_size, size_t beam_width, size_t max_seq_len, size_t max_cache_seq_len, size_t max_input_len)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    const size_t batchxbeam      = batch_size * beam_width;
    const size_t self_cache_size = (num_layer_ / pipeline_para_.world_size_) * batchxbeam * max_cache_seq_len
                                   * hidden_units_ / tensor_para_.world_size_;

    if (vocab_size_ != vocab_size_padded_) {
        padded_embedding_kernel_ =
            (T*)(allocator_->reMalloc(padded_embedding_kernel_, sizeof(T) * hidden_units_ * vocab_size_padded_, true));
        padded_embedding_kernel_ptr_ = padded_embedding_kernel_;
    }

    input_attention_mask_ = (T*)(allocator_->reMalloc(
        input_attention_mask_, sizeof(T) * batchxbeam * max_seq_len * max_cache_seq_len, false));
    decoder_input_buf_ = (T*)(allocator_->reMalloc(decoder_input_buf_, sizeof(T) * batchxbeam * hidden_units_, false));
    decoder_output_buf_ =
        (T*)(allocator_->reMalloc(decoder_output_buf_, sizeof(T) * batchxbeam * hidden_units_, false));

    key_cache_   = (T*)(allocator_->reMalloc(key_cache_, sizeof(T) * self_cache_size * 2, true));
    value_cache_ = key_cache_ + self_cache_size;

    // prompt_learning weight batch ptrs
    // prompt_learning_weight_batch_ =
    //     (const T**)(allocator_->reMalloc(prompt_learning_weight_batch_, sizeof(T*) * batchxbeam, false));
    tiled_prompt_lengths_buf_ =
        (int*)(allocator_->reMalloc(tiled_prompt_lengths_buf_, sizeof(int) * batchxbeam, false));

    tiled_input_ids_buf_ =
        (int*)(allocator_->reMalloc(tiled_input_ids_buf_, sizeof(int) * batchxbeam * max_input_len, true));
    tiled_input_lengths_buf_ = (int*)(allocator_->reMalloc(tiled_input_lengths_buf_, sizeof(int) * batchxbeam, true));

    context_decoder_input_buf_  = (T*)(allocator_->reMalloc(
        context_decoder_input_buf_, sizeof(T) * batchxbeam * max_input_len * hidden_units_, false));
    context_decoder_output_buf_ = (T*)(allocator_->reMalloc(
        context_decoder_output_buf_, sizeof(T) * batchxbeam * max_input_len * hidden_units_, false));
    normed_context_decoder_output_buf_ = (T*)(allocator_->reMalloc(
        normed_context_decoder_output_buf_, sizeof(T) * batchxbeam * max_input_len * hidden_units_, false));

    is_allocate_buffer_ = true;
}

template<typename T>
void Llama<T>::freeBuffer()
{
    if (is_allocate_buffer_) {
        if (vocab_size_ != vocab_size_padded_) {
            padded_embedding_kernel_ptr_ = nullptr;
            allocator_->free((void**)(&padded_embedding_kernel_));
        }

        allocator_->free((void**)(&input_attention_mask_));
        allocator_->free((void**)(&decoder_input_buf_));
        allocator_->free((void**)(&decoder_output_buf_));
        allocator_->free((void**)(&key_cache_));
        // allocator_->free((void**)(&prompt_learning_weight_batch_));
        allocator_->free((void**)(&tiled_prompt_lengths_buf_));
        allocator_->free((void**)(&tiled_input_ids_buf_));
        allocator_->free((void**)(&tiled_input_lengths_buf_));
        allocator_->free((void**)(&context_decoder_input_buf_));
        allocator_->free((void**)(&context_decoder_output_buf_));
        allocator_->free((void**)(&normed_context_decoder_output_buf_));

        is_allocate_buffer_ = false;
    }
}

template<typename T>
Llama<T>::Llama(size_t                              head_num,
                size_t                              size_per_head,
                size_t                              inter_size,
                size_t                              num_layer,
                size_t                              vocab_size,
                size_t                              rotary_embedding_dim,
                float                               layernorm_eps,
                int                                 start_id,
                int                                 end_id,
                int                                 prompt_learning_start_id,  // only needed by p/prompt-tuning
                PromptLearningType                  prompt_learning_type,
                bool                                use_gptj_residual,
                float                               beam_search_diversity_rate,
                size_t                              top_k,
                float                               top_p,
                unsigned long long                  random_seed,
                float                               temperature,
                float                               len_penalty,
                float                               repetition_penalty,
                cudaStream_t                        stream,
                cublasMMWrapper*                    cublas_wrapper,
                IAllocator*                         allocator,
                bool                                is_free_buffer_after_forward,
                cudaDeviceProp*                     cuda_device_prop,
                AttentionType                       attention_type,
                std::shared_ptr<AbstractCustomComm> custom_all_reduce_comm,
                int                                 enable_custom_all_reduce):
    BaseLayer(stream, cublas_wrapper, allocator, is_free_buffer_after_forward, cuda_device_prop),
    head_num_(head_num),
    size_per_head_(size_per_head),
    inter_size_(inter_size),
    num_layer_(num_layer),
    vocab_size_(vocab_size),
    rotary_embedding_dim_(rotary_embedding_dim),
    layernorm_eps_(layernorm_eps),
    start_id_(start_id),
    end_id_(end_id),
    prompt_learning_start_id_(prompt_learning_start_id),
    prompt_learning_type_(prompt_learning_type),
    use_gptj_residual_(use_gptj_residual),
    hidden_units_(head_num * size_per_head),
    local_head_num_(head_num / 1),
    attention_type_(attention_type)
{
    tensor_para_.world_size_   = 1;
    tensor_para_.rank_         = 0;
    pipeline_para_.world_size_ = 1;
    pipeline_para_.rank_       = 0;

    int local_vacab_size = ceil(vocab_size_ / 1.f / tensor_para_.world_size_);
    if (std::is_same<half, T>::value
#ifdef ENABLE_BF16
        || std::is_same<__nv_bfloat16, T>::value
#endif
    ) {
        local_vacab_size = ceil(local_vacab_size / 8.f) * 8;
    }
    vocab_size_padded_ = (size_t)local_vacab_size * tensor_para_.world_size_;
    initialize();
}

template<typename T>
Llama<T>::Llama(size_t                              head_num,
                size_t                              size_per_head,
                size_t                              inter_size,
                size_t                              num_layer,
                size_t                              vocab_size,
                size_t                              rotary_embedding_dim,
                float                               layernorm_eps,
                int                                 start_id,
                int                                 end_id,
                int                                 prompt_learning_start_id,  // only needed by p/prompt-tuning
                PromptLearningType                  prompt_learning_type,
                bool                                use_gptj_residual,
                float                               beam_search_diversity_rate,
                size_t                              top_k,
                float                               top_p,
                unsigned long long                  random_seed,
                float                               temperature,
                float                               len_penalty,
                float                               repetition_penalty,
                NcclParam                           tensor_para,
                NcclParam                           pipeline_para,
                cudaStream_t                        stream,
                cublasMMWrapper*                    cublas_wrapper,
                IAllocator*                         allocator,
                bool                                is_free_buffer_after_forward,
                cudaDeviceProp*                     cuda_device_prop,
                AttentionType                       attention_type,
                std::shared_ptr<AbstractCustomComm> custom_all_reduce_comm,
                int                                 enable_custom_all_reduce):
    BaseLayer(stream, cublas_wrapper, allocator, is_free_buffer_after_forward, cuda_device_prop),
    head_num_(head_num),
    size_per_head_(size_per_head),
    inter_size_(inter_size),
    num_layer_(num_layer),
    vocab_size_(vocab_size),
    rotary_embedding_dim_(rotary_embedding_dim),
    layernorm_eps_(layernorm_eps),
    start_id_(start_id),
    end_id_(end_id),
    prompt_learning_start_id_(prompt_learning_start_id),
    prompt_learning_type_(prompt_learning_type),
    use_gptj_residual_(use_gptj_residual),
    hidden_units_(head_num * size_per_head),
    tensor_para_(tensor_para),
    pipeline_para_(pipeline_para),
    local_head_num_(head_num / tensor_para.world_size_),
    custom_all_reduce_comm_(custom_all_reduce_comm),
    enable_custom_all_reduce_(enable_custom_all_reduce),
    attention_type_(attention_type)
{
    int local_vacab_size = ceil(vocab_size_ / 1.f / tensor_para_.world_size_);
    if (std::is_same<half, T>::value) {
        local_vacab_size = ceil(local_vacab_size / 8.f) * 8;
    }
    vocab_size_padded_ = (size_t)local_vacab_size * tensor_para_.world_size_;
    initialize();
}

template<typename T>
Llama<T>::Llama(Llama<T> const& gpt):
    BaseLayer(gpt),
    head_num_(gpt.head_num_),
    size_per_head_(gpt.size_per_head_),
    inter_size_(gpt.inter_size_),
    num_layer_(gpt.num_layer_),
    vocab_size_(gpt.vocab_size_),
    rotary_embedding_dim_(gpt.rotary_embedding_dim_),
    layernorm_eps_(gpt.layernorm_eps_),
    start_id_(gpt.start_id_),
    end_id_(gpt.end_id_),
    prompt_learning_start_id_(gpt.prompt_learning_start_id_),
    prompt_learning_type_(gpt.prompt_learning_type_),
    use_gptj_residual_(gpt.use_gptj_residual_),
    hidden_units_(gpt.hidden_units_),
    tensor_para_(gpt.tensor_para_),
    pipeline_para_(gpt.pipeline_para_),
    local_head_num_(gpt.local_head_num_),
    vocab_size_padded_(gpt.vocab_size_padded_),
    custom_all_reduce_comm_(gpt.custom_all_reduce_comm_),
    enable_custom_all_reduce_(gpt.enable_custom_all_reduce_),
    attention_type_(gpt.attention_type_)
{
    initialize();
}

template<typename T>
Llama<T>::~Llama()
{
    delete gpt_context_decoder_;
    freeBuffer();
}

template<typename T>
void Llama<T>::registerCallback(callback_sig* fn, void* ctx)
{
    token_generated_cb_  = fn;
    token_generated_ctx_ = ctx;
}

template<typename T>
void Llama<T>::unRegisterCallback()
{
    token_generated_cb_  = nullptr;
    token_generated_ctx_ = nullptr;
}

template<typename T>
void Llama<T>::forward(std::vector<Tensor>*       output_tensors,
                         const std::vector<Tensor>* input_tensors,
                         const LlamaWeight<T>*    gpt_weights)
{
    FT_CHECK(false);
}

template<typename T>
void Llama<T>::forward(std::unordered_map<std::string, Tensor>*       output_tensors,
                         const std::unordered_map<std::string, Tensor>* input_tensors,
                         const LlamaWeight<T>*                        gpt_weights)
{
    // input_tensors:
    //      input_ids [batch_size, max_input_length]
    //      input_lengths [batch_size]
    //      prompt_learning_task_name_ids [batch_size] on cpu, optional
    //      output_seq_len [batch_size] on cpu
    //      start_id [batch_size] on cpu, optional
    //      end_id [batch_size] on cpu, optional
    //      stop_words_list [batch_size, 2, stop_words_length], optional
    //      bad_words_list [2, bad_words_length] or [batch_size, 2, bad_words_length], optional
    //      runtime_top_k [1] or [batch_size] on cpu, optional, uint.
    //      runtime_top_p [1] or [batch_size] on cpu, optional, float.
    //      beam_search_diversity_rate [1] or [batch_size] on cpu, optional, float.
    //      temperature [1] or [batch_size] on cpu, optional, float.
    //      len_penalty [1] or [batch_size] on cpu, optional, float.
    //      repetition_penalty [1] or [batch_size] on cpu, optional, float.
    //      min_length [1] or [batch_size] on cpu, optional, int
    //      random_seed [1] or [batch_size] on cpu, optional, unsigned long long int.
    //      request_prompt_lengths [batch_size], optional
    //      request_prompt_embedding [batch_size, max_prompt_length, hidden_units], float, optional
    //      requst_prompt_type [batch_size], int, optional
    //      top_p_decay [batch_size] on gpu, float, optional
    //      top_p_min [batch_size] on gpu, float, optional
    //      top_p_reset_ids [batch_size] on gpu, uint32, optional

    // output_tensors:
    //      output_ids [batch_size, beam_width, max_output_seq_len]
    //      sequence_length [batch_size, beam_width]
    //      output_log_probs [batch_size, beam_width, request_output_seq_len], must be float*.
    //          optional. It leads to additional computing cost. If we don't need this result, don't put it.
    //      cum_log_probs [batch_size, beam], optional, must be float*.
    //          optional. It leads to additional computing cost. If we don't need this result, don't put it.

    // Step is from max_input_length ~ max_output_seq_len,
    // When step = k,  we put output ids and caches at step k, and the sequence_length would be k - 1 before
    // complete this step.
    // When there is no input_ids, put the start token at step 0 of output_ids_buf_. After forward, only copy
    // the step 1 ~ max_output_seq_len of output_ids_buf_ to output_tensors->at(0).data

    FT_CHECK_WITH_INFO(input_tensors->size() >= 3, "input_tensors->size() >= 3");
    FT_CHECK_WITH_INFO(output_tensors->size() >= 2, "output_tensors->size() >= 2");
    FT_CHECK(input_tensors->at("input_ids").shape.size() == 2);
    FT_CHECK(input_tensors->at("input_lengths").shape.size() == 1);
    FT_CHECK(input_tensors->find("output_seq_len") != input_tensors->end()
             && input_tensors->at("output_seq_len").shape.size() == 1);
    FT_CHECK(output_tensors->at("output_ids").shape.size() == 3);
    FT_CHECK_WITH_INFO(input_tensors->at("input_ids").shape[0] == output_tensors->at("output_ids").shape[0],
                       "input_tensors->at(\"input_ids\").shape[0] == output_tensors->at(\"output_ids\").shape[0]");

    const size_t batch_size = output_tensors->at("output_ids").shape[0];
    const size_t beam_width = output_tensors->at("output_ids").shape[1];

    PromptLearningType request_prompt_type = PromptLearningType::no_prompt;
    int                valid_prompt_inputs = input_tensors->count("request_prompt_type")
                              + input_tensors->count("request_prompt_lengths")
                              + input_tensors->count("request_prompt_embedding");

    if (valid_prompt_inputs == 3) {
        request_prompt_type = static_cast<PromptLearningType>(input_tensors->at("request_prompt_type").getVal<int>());
        FT_LOG_INFO("Apply prompt embedding from input, will ignore task name ids");
    }
    else if (valid_prompt_inputs > 0) {
        FT_LOG_WARNING(
            "Prompts not applied: request_prompt_embedding, request_prompt_lengths, request_prompt_type are all needed!");
    }
    if (request_prompt_type == PromptLearningType::prefix_prompt) {
        FT_LOG_WARNING("Request prompt doesn't support prefix prompt currently!");
    }

    // Prefix Prompt Inputs
    // Padding works as follows: p p x x i i i x x --> p p i i i x x x x (p denotes prompt, i denotes input, x denotes
    // pad)
    // TODO (perkzz): move unnecessary paddings
    // const int* prompt_learning_task_name_ids =
    //     input_tensors->count("prompt_learning_task_name_ids") ?
    //         input_tensors->at("prompt_learning_task_name_ids").getPtr<const int>() :
    //         nullptr;
    // has_prefix_prompt_ =
    //     (prompt_learning_task_name_ids != nullptr) && (prompt_learning_type_ == PromptLearningType::prefix_prompt);
    // int max_prefix_prompt_length = 0;

    // FT_CHECK_WITH_INFO(
    //     !(prompt_learning_task_name_ids != nullptr
    //       && (prompt_learning_type_ == PromptLearningType::no_prompt
    //           || prompt_learning_type_ == PromptLearningType::soft_prompt)),
    //     "prompt_learning_type is prefix_prompt either p_prompt_tuning when prompt_learning_task_name_ids are provided.");

    // NOTE: Prefix Prompt PreProcessing
    // get prefix_prompt_weight for each batch --> shape [batch, beam_width]
    // --> ptrs with shape [num_layers, 2, num_heads, perfix_seq_len, size_per_head]
    // std::vector<const T*> prefix_prompt_weight_batch_ptrs;
    // std::vector<int>      prefix_prompt_lengths;
    // if (has_prefix_prompt_) {
    //     for (int bs_id = 0; bs_id < batch_size; ++bs_id) {
    //         int task_id = prompt_learning_task_name_ids[bs_id];
    //         // throw errors when prompt task_name_ids are not found
    //         std::pair<const T*, int> prefix_prompt_weight_length_pair;
    //         try {
    //             prefix_prompt_weight_length_pair = gpt_weights->prompt_learning_table.at(task_id);
    //         }
    //         catch (const std::out_of_range& oor) {
    //             FT_LOG_ERROR("prefix_prompt_weights_lengths not found for prompt task id: " + task_id);
    //             throw oor;
    //         }
    //         for (int bw_id = 0; bw_id < beam_width; ++bw_id) {
    //             prefix_prompt_weight_batch_ptrs.push_back(prefix_prompt_weight_length_pair.first);
    //             prefix_prompt_lengths.push_back(prefix_prompt_weight_length_pair.second);
    //         }
    //     }

    //     max_prefix_prompt_length = *max_element(prefix_prompt_lengths.begin(), prefix_prompt_lengths.end());

    //     FT_LOG_DEBUG("max_prefix_prompt_length: %d", max_prefix_prompt_length);

    //     if (max_prefix_prompt_length == 0) {
    //         has_prefix_prompt_ = false;
    //         FT_LOG_DEBUG("prompts are not applied !");
    //     }
    // }

    int max_input_length = input_tensors->at("input_ids").shape[1];
    // FT_CHECK_WITH_INFO(!(max_input_length == 0 && max_prefix_prompt_length > 0),
    //                    "Prefix Prompt should come with inputs!");

    // Prefix Soft Prompt
    has_prefix_soft_prompt_ = request_prompt_type == PromptLearningType::soft_prompt;
    const size_t max_prefix_soft_prompt_length =
        has_prefix_soft_prompt_ ? input_tensors->at("request_prompt_embedding").shape[1] : 0;
    const size_t limit_len_offset   = max_prefix_soft_prompt_length + (max_input_length == 0 ? 1 : 0);
    const size_t max_output_seq_len = input_tensors->at("output_seq_len").max<uint32_t>() + limit_len_offset;
    const size_t max_seq_len        = max_output_seq_len;
    // max cache seq len should include max prefix prompt length as it has k/v states
    const size_t max_cache_seq_len = max_output_seq_len; /*max_prefix_prompt_length;*/
    if (max_cache_seq_len < max_seq_len) {
        FT_LOG_WARNING("max_cache_seq_len (%d) is less than max_seq_len (%d). "
                       "Note that this reduces the memory cost of k/v cache, but may hurt the accuracy.",
                       max_cache_seq_len,
                       max_seq_len);
    }
    else if (max_cache_seq_len > max_seq_len) {
        FT_LOG_WARNING("max_cache_seq_len (%d) is larger than max_seq_len (%d). "
                       "This may lead to additional memory cost. Suggest to use smaller max_cache_seq_len.",
                       max_cache_seq_len,
                       max_seq_len);
    }
    const cudaDataType_t gemm_data_type = getCudaDataType<T>();
    allocateBuffer(
        batch_size, beam_width, max_seq_len, max_cache_seq_len, max_input_length + max_prefix_soft_prompt_length);

    sync_check_cuda_error();
    
    const DataType data_type = getTensorType<T>();

    const std::vector<size_t> self_k_cache_shape = {num_layer_ / pipeline_para_.world_size_,
                                                    batch_size * beam_width,
                                                    local_head_num_,
                                                    size_per_head_ / (16 / sizeof(T)),
                                                    max_cache_seq_len,
                                                    16 / sizeof(T)};
    const std::vector<size_t> self_v_cache_shape = {num_layer_ / pipeline_para_.world_size_,
                                                    batch_size * beam_width,
                                                    local_head_num_,
                                                    max_cache_seq_len,
                                                    size_per_head_};

    // Prefix prompts
    // if (has_prefix_prompt_) {
    //     cudaMemcpyAsync(prompt_learning_weight_batch_,
    //                     prefix_prompt_weight_batch_ptrs.data(),
    //                     sizeof(T*) * batch_size * beam_width,
    //                     cudaMemcpyDefault,
    //                     stream_);
    //     cudaMemcpyAsync(tiled_prompt_lengths_buf_,
    //                     prefix_prompt_lengths.data(),
    //                     sizeof(int) * batch_size * beam_width,
    //                     cudaMemcpyDefault,
    //                     stream_);
    // }

    sync_check_cuda_error();

    // handle first step
    if (has_prefix_soft_prompt_ || max_input_length > 1) {
        invokeTileGptInputs(tiled_input_ids_buf_,
                            tiled_input_lengths_buf_,
                            input_tensors->at("input_ids").getPtr<int>(),
                            input_tensors->at("input_lengths").getPtr<const int>(),
                            batch_size,
                            beam_width,
                            max_input_length,
                            stream_);
        sync_check_cuda_error();
        
        if (pipeline_para_.rank_ == 0) {
            invokeInputIdsEmbeddingLookupPosEncoding(context_decoder_input_buf_,
                                                    nullptr,
                                                    gpt_weights->pre_decoder_embedding_table,
                                                    gpt_weights->position_encoding_table,
                                                    pPromptTuningParam<T>{},  // no p/prompt tuning
                                                    tiled_input_ids_buf_,
                                                    1,
                                                    max_input_length,
                                                    max_input_length,
                                                    batch_size * beam_width,
                                                    hidden_units_,
                                                    stream_);
        }
        // sync_check_cuda_error();

        invokeBuildDecoderAttentionMask(input_attention_mask_,
                                        tiled_input_lengths_buf_,
                                        tiled_prompt_lengths_buf_,
                                        batch_size * beam_width,
                                        max_input_length,
                                        0, //max_prefix_prompt_length,
                                        stream_);
        sync_check_cuda_error();

        std::unordered_map<std::string, Tensor> decoder_input_tensors{
            {"decoder_input",
             Tensor{MEMORY_GPU,
                    data_type,
                    {batch_size * beam_width, (size_t)max_input_length, hidden_units_},
                    context_decoder_input_buf_}},
            {"attention_mask",
             Tensor{MEMORY_GPU,
                    data_type,
                    {batch_size * beam_width,
                     1,
                     (size_t)max_input_length,
                     (size_t)(max_input_length /*+ max_prefix_prompt_length*/)},
                    input_attention_mask_}},
            {"input_lengths", Tensor{MEMORY_GPU, TYPE_INT32, {batch_size * beam_width}, tiled_input_lengths_buf_}},
            {"d_prefix_prompt_batch",
             Tensor{MEMORY_GPU,
                    data_type,
                    {batch_size * beam_width},
                    /* has_prefix_prompt_ ? prompt_learning_weight_batch_ : */nullptr}},
            {"d_prefix_prompt_lengths",
             Tensor{MEMORY_GPU,
                    TYPE_INT32,
                    {batch_size * beam_width},
                    /* has_prefix_prompt_ ? tiled_prompt_lengths_buf_ : */nullptr}}};

        std::unordered_map<std::string, Tensor> decoder_output_tensors{
            {"decoder_output",
             Tensor{MEMORY_GPU,
                    data_type,
                    {batch_size * beam_width, (size_t)max_input_length, hidden_units_},
                    context_decoder_output_buf_}},
            {"key_cache", Tensor{MEMORY_GPU, data_type, self_k_cache_shape, key_cache_}},
            {"value_cache", Tensor{MEMORY_GPU, data_type, self_v_cache_shape, value_cache_}},
            {"last_token_hidden_units",
             Tensor{MEMORY_GPU, data_type, {batch_size * beam_width, hidden_units_}, decoder_output_buf_}}};

        gpt_context_decoder_->forward(
            &decoder_output_tensors, &decoder_input_tensors, &gpt_weights->decoder_layer_weights);
        sync_check_cuda_error();
    }
   
    if (pipeline_para_.rank_ == pipeline_para_.world_size_ - 1) {
        if (vocab_size_ == vocab_size_padded_) {
            padded_embedding_kernel_ptr_ = gpt_weights->post_decoder_embedding.kernel;
        }

        // LlamaRMSNorm()
        invokeGeneralT5LayerNorm(normed_context_decoder_output_buf_,   // output
                            context_decoder_output_buf_,               // input
                            gpt_weights->post_decoder_layernorm.gamma, // weight
                            (const T*)nullptr,                         // beta, we don't need
                            layernorm_eps_, 
                            batch_size * beam_width * max_input_length,// m
                            hidden_units_,                             // n
                            stream_);

        // lm_head linear
        // Expected output tensor size is (bs, seq_len, vocab_size)
        cublas_wrapper_->Gemm(CUBLAS_OP_T,
                                CUBLAS_OP_N,
                                vocab_size_padded_,  // m = output row
                                batch_size * beam_width * max_input_length, // n = output col
                                hidden_units_,  // k = common dimension
                                padded_embedding_kernel_ptr_,
                                hidden_units_,  // k
                                normed_context_decoder_output_buf_,
                                hidden_units_,
                                output_tensors->at("logits_buf").getPtr<half>(),
                                vocab_size_padded_
                                );
        // Print tenser for debugging
        // for (int bs = 0; bs < batch_size; bs++) {
        //     for(int i = max_input_length-2; i<max_input_length; i++) {
        //         FT_LOG_INFO("\nbs %d, seq %d first 128 elements", bs, i);
        //         print_to_screen(lm_head_context_decoder_output_buf_ + vocab_size_padded_*(bs*max_input_length + i), 128);
        //     }
        // }
    }
}

template<typename T>
void Llama<T>::sendTensorsToFirstPipelineNode(std::unordered_map<std::string, Tensor>*       output_tensors,
                                                const std::unordered_map<std::string, Tensor>* input_tensors)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    if (pipeline_para_.world_size_ == 1) {
        // throw errors when detected
        ftNcclStreamSynchronize(tensor_para_, pipeline_para_, stream_);
        return;
    }

    const auto pp_rank = pipeline_para_.rank_;

    ftNcclGroupStart();
    for (auto const& it : *output_tensors) {
        if (it.second.data == nullptr) {
            continue;
        }

        if (pp_rank == pipeline_para_.world_size_ - 1) {
            ftNcclSend(it.second.getPtr<char>(), it.second.sizeBytes(), 0, pipeline_para_, stream_);
        }
        else if (pp_rank == 0) {
            ftNcclRecv(it.second.getPtr<char>(),
                       it.second.sizeBytes(),
                       pipeline_para_.world_size_ - 1,
                       pipeline_para_,
                       stream_);
        }
    }
    ftNcclGroupEnd();
    // throw errors when detected
    ftNcclStreamSynchronize(tensor_para_, pipeline_para_, stream_);
}

template<typename T>
size_t Llama<T>::getPipelineParallelRank()
{
    return pipeline_para_.rank_;
}

template<typename T>
size_t Llama<T>::getPipelineParallelSize()
{
    return pipeline_para_.world_size_;
}

template<typename T>
size_t Llama<T>::getTensorParallelRank()
{
    return tensor_para_.rank_;
}

template<typename T>
size_t Llama<T>::getTensorParallelSize()
{
    return tensor_para_.world_size_;
}

template class Llama<float>;
template class Llama<half>;
#ifdef ENABLE_BF16
template class Llama<__nv_bfloat16>;
#endif

}  // namespace fastertransformer
