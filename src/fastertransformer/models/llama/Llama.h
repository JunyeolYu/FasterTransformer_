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

#pragma once

#include <cstddef>
#include <vector>

#include "src/fastertransformer/layers/DynamicDecodeLayer.h"
#include "src/fastertransformer/models/llama/LlamaContextDecoder.h"
#include "src/fastertransformer/models/llama/LlamaDecoder.h"
#include "src/fastertransformer/models/llama/LlamaWeight.h"
#include "src/fastertransformer/utils/custom_ar_comm.h"
#include "src/fastertransformer/utils/prompt_learning.h"
#include "src/fastertransformer/utils/logger.h"

namespace fastertransformer {

template<typename T>
class Llama: public BaseLayer {
private:
    // meta data
    size_t head_num_;
    size_t size_per_head_;
    size_t inter_size_;
    size_t num_layer_;
    size_t vocab_size_;
    size_t rotary_embedding_dim_;
    float layernorm_eps_;

    static constexpr bool  neox_rotary_style_ = true;

    int    start_id_;
    int    end_id_;
    size_t hidden_units_;

    size_t    local_head_num_;
    NcclParam tensor_para_;
    NcclParam pipeline_para_;

    std::shared_ptr<AbstractCustomComm> custom_all_reduce_comm_;
    int                                 enable_custom_all_reduce_;

    AttentionType attention_type_;

    size_t     vocab_size_padded_;
    const bool is_context_qk_buf_float_ =
        (std::getenv("CONTEXT_ATTENTION_BMM1_HALF_ACCUM") == nullptr ||
         std::string(std::getenv("CONTEXT_ATTENTION_BMM1_HALF_ACCUM")) != "ON");

    // Residual Type
    const bool use_gptj_residual_ = false;

    // Prompt Learning Parameters
    PromptLearningType prompt_learning_type_;
    int                prompt_learning_start_id_;  // start_id for prompt_learning (only needed by prefix prompts)
    bool               has_prefix_prompt_;
    bool               has_prefix_soft_prompt_;

    LlamaContextDecoder<T>*  gpt_context_decoder_;

    void allocateBuffer() override;
    void allocateBuffer(
        size_t batch_size, size_t beam_width, size_t max_seq_len, size_t max_cache_seq_len, size_t max_input_len);
    void freeBuffer() override;

    void initialize();

protected:
    T*       padded_embedding_kernel_;
    const T* padded_embedding_kernel_ptr_;

    T* input_attention_mask_;

    T* decoder_input_buf_;
    T* decoder_output_buf_;

    // T*   key_cache_;
    // T*   value_cache_;

    // prompt_learning weight_batch ptrs
    // const T** prompt_learning_weight_batch_;
    int*      tiled_prompt_lengths_buf_;  // only needed by prefix prompts

    int*  tiled_input_ids_buf_;
    int*  tiled_input_lengths_buf_;

    T*     context_decoder_input_buf_;
    T*     context_decoder_output_buf_;
    T*     normed_context_decoder_output_buf_;

    // function pointer callback
    using callback_sig                 = void(std::unordered_map<std::string, Tensor>*, void*);
    callback_sig* token_generated_cb_  = nullptr;
    void*         token_generated_ctx_ = nullptr;

    // callback step
    size_t token_generated_cb_step_ = 5; // default 5, override by env LLAMA_STREAM_CB_STEP

    void sendTensorsToFirstPipelineNode(std::unordered_map<std::string, Tensor>*       output_tensors,
                                        const std::unordered_map<std::string, Tensor>* input_tensors);

public:
    Llama(size_t                              head_num,
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
          cudaDeviceProp*                     cuda_device_prop         = nullptr,
          AttentionType                       attention_type           = AttentionType::UNFUSED_MHA,
          std::shared_ptr<AbstractCustomComm> custom_all_reduce_comm   = nullptr,
          int                                 enable_custom_all_reduce = 0);

    Llama(size_t                              head_num,
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
          cudaDeviceProp*                     cuda_device_prop         = nullptr,
          AttentionType                       attention_type           = AttentionType::UNFUSED_MHA,
          std::shared_ptr<AbstractCustomComm> custom_all_reduce_comm   = nullptr,
          int                                 enable_custom_all_reduce = 0);

    Llama(Llama<T> const& Llama);

    ~Llama();

    void forward(std::vector<Tensor>*       output_tensors,
                 const std::vector<Tensor>* input_tensors,
                 const LlamaWeight<T>*      gpt_weights);

    void forward(std::unordered_map<std::string, Tensor>*       output_tensors,
                 const std::unordered_map<std::string, Tensor>* input_tensors,
                 const LlamaWeight<T>*                          gpt_weights);

    size_t getPipelineParallelRank();
    size_t getPipelineParallelSize();
    size_t getTensorParallelRank();
    size_t getTensorParallelSize();

    void registerCallback(callback_sig* fn, void* ctx);
    void unRegisterCallback();
};

}  // namespace fastertransformer
